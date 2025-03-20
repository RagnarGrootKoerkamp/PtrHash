// TODO:
// - Specialization for instances with a single part.
// - Use trace instead of eprintln.
#![cfg_attr(feature = "unstable", feature(iter_array_chunks))]
//! PTRHash is a minimal perfect hash function.
//!
//! Usage example:
//! ```rust
//! use ptr_hash::{PtrHash, PtrHashParams};
//!
//! // Generate some random keys.
//! let n = 1_000_000_000;
//! let keys = ptr_hash::util::generate_keys(n);
//!
//! // Build the datastructure.
//! let mphf = <PtrHash>::new(&keys, PtrHashParams::default());
//!
//! // Get the minimal index of a key.
//! let key = 0;
//! let idx = mphf.index_minimal(&key);
//! assert!(idx < n);
//!
//! // Get the non-minimal index of a key. Slightly faster.
//! let _idx = mphf.index(&key);
//!
//! // An iterator over the indices of the keys.
//! // 32: number of iterations ahead to prefetch.
//! // true: remap to a minimal key in [0, n).
//! let indices = mphf.index_stream::<32, true>(&keys);
//! assert_eq!(indices.sum::<usize>(), (n * (n - 1)) / 2);
//!
//! // Test that all items map to different indices
//! let mut taken = vec![false; n];
//! for key in keys {
//!     let idx = mphf.index_minimal(&key);
//!     assert!(!taken[idx]);
//!     taken[idx] = true;
//! }
//! ```
//#![cfg_attr(target_arch = "aarch64", feature(stdsimd))]
#![allow(clippy::needless_range_loop)]

/// Customizable Hasher trait.
pub mod hash;
/// Extendable backing storage trait and types.
pub mod pack;
/// Some internal logging and testing utilities.
pub mod util;

pub mod bucket_fn;
mod bucket_idx;
mod build;
mod reduce;
mod shard;
mod sort_buckets;
pub mod stats;
#[cfg(test)]
mod test;

use bitvec::{bitvec, vec::BitVec};
use bucket_fn::BucketFn;
pub use bucket_fn::CubicEps;
pub use bucket_fn::SquareEps;
pub use bucket_fn::Linear;
pub use cacheline_ef::CachelineEfVec;
use itertools::izip;
use itertools::Itertools;
use mem_dbg::MemSize;
use pack::EliasFano;
use pack::MutPacked;
use rand::{random, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rdst::RadixSort;
pub use shard::Sharding;
use stats::BucketStats;
use std::{borrow::Borrow, default::Default, marker::PhantomData, time::Instant};

use crate::{hash::*, pack::Packed, reduce::*, util::log_duration};

/// Parameters for PtrHash construction.
///
/// Since these are not used in inner loops they are simple variables instead of template arguments.
#[derive(Clone, Copy, Debug, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct PtrHashParams<BF> {
    /// Set to false to disable remapping to a minimal PHF.
    pub remap: bool,
    /// Use `n/alpha` slots approximately.
    pub alpha: f64,
    /// Use average bucket size lambda.
    pub lambda: f64,
    /// Bucket function
    pub bucket_fn: BF,
    /// If given, #slots/part will be the smallest power of 2 at least this.
    /// By default, it is computed as the smallest power of 2 for which construction is likely to succeed.
    pub slots_per_part: Option<usize>,
    /// Upper bound on number of keys per shard.
    /// Default is 2^32, or 32GB of hashes per shard.
    pub keys_per_shard: usize,
    /// When true, write each shard to a file instead of iterating multiple
    /// times.
    pub sharding: Sharding,

    /// Print bucket size and pilot stats after construction.
    pub print_stats: bool,
}

impl PtrHashParams<Linear> {
    pub fn default_fast() -> Self {
        Self {
            remap: true,
            alpha: 0.99,
            lambda: 3.0,
            bucket_fn: Linear,
            slots_per_part: None,
            // By default, limit to 2^32 keys per shard, whose hashes take 8B*2^31=16GB.
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
            print_stats: false,
        }
    }
}

impl PtrHashParams<SquareEps> {
    pub fn default_square() -> Self {
        Self {
            remap: true,
            alpha: 0.99,
            lambda: 3.5,
            bucket_fn: SquareEps,
            slots_per_part: None,
            // By default, limit to 2^32 keys per shard, whose hashes take 8B*2^31=16GB.
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
            print_stats: false,
        }
    }
}

impl PtrHashParams<CubicEps> {
    pub fn default_compact() -> Self {
        Self {
            remap: true,
            alpha: 0.99,
            lambda: 4.0,
            bucket_fn: CubicEps,
            slots_per_part: None,
            // By default, limit to 2^32 keys per shard, whose hashes take 8B*2^31=16GB.
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
            print_stats: false,
        }
    }
}

impl Default for PtrHashParams<CubicEps> {
    /// Default 'compact' parameters:
    /// - `alpha=0.99`
    /// - `lambda=3.5`
    /// - `bucket_fn=CubicEps`
    fn default() -> Self {
        Self {
            remap: true,
            alpha: 0.99,
            lambda: 3.5,
            bucket_fn: CubicEps,
            slots_per_part: None,
            // By default, limit to 2^32 keys per shard, whose hashes take 8B*2^31=16GB.
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
            print_stats: false,
        }
    }
}

// Externally visible aliases for convenience.

/// An alias for PtrHash with default generic arguments.
/// Using this, you can write `DefaultPtrHash::new()` instead of `<PtrHash>::new()`.
pub type DefaultPtrHash<H, Key, BF> = PtrHash<Key, BF, CachelineEfVec, H, Vec<u8>>;

/// Using EliasFano for the remap is slower but uses slightly less memory.
pub type EfPtrHash<H, Key> = PtrHash<Key, CubicEps, EliasFano, H, Vec<u8>>;

/// Trait that keys must satisfy.
pub trait KeyT: Send + Sync + std::hash::Hash {}
impl<T: Send + Sync + std::hash::Hash> KeyT for T {}

// Some fixed algorithmic decisions.
type Rp = FastReduce;
type Rb = FastReduce;
type Rs = MulReduce;
type Pilot = u64;
type PilotHash = u64;

/// PtrHash datastructure.
/// The recommended way to use PtrHash with default types.
///
/// `F`: The packing to use for remapping free slots, default `TinyEf`.
/// `Hx`: The hasher to use for keys, default `FxHash`.
/// `V`: The pilots type. Usually `Vec<u8>`, or `&[u8]` for Epserde.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone, MemSize)]
pub struct PtrHash<
    Key: KeyT + ?Sized = u64,
    BF: BucketFn = bucket_fn::CubicEps,
    F: Packed = CachelineEfVec,
    Hx: Hasher<Key> = hash::FxHash,
    V: AsRef<[u8]> = Vec<u8>,
> {
    params: PtrHashParams<BF>,

    /// The number of keys.
    n: usize,
    /// The total number of parts.
    parts: usize,
    /// The number of shards.
    shards: usize,
    /// The maximal number of parts per shard.
    /// The last shard may have fewer parts.
    parts_per_shard: usize,
    /// The total number of slots.
    slots_total: usize,
    /// The total number of buckets.
    buckets_total: usize,
    /// The number of slots per part, always a power of 2.
    slots: usize,
    /// Since s is a power of 2, we can compute multiplications using a shift
    /// instead.
    lg_slots: u32,
    /// The number of buckets per part.
    buckets: usize,

    // Precomputed fast modulo operations.
    /// Fast %shards.
    rem_shards: Rp,
    /// Fast %parts.
    rem_parts: Rp,
    /// Fast &b.
    rem_buckets: Rb,
    /// Fast &b_total.
    rem_buckets_total: Rb,

    /// Fast %s.
    rem_slots: Rs,

    // Computed state.
    /// The global seed.
    seed: u64,
    /// The pilots.
    pilots: V,
    /// Remap the out-of-bound slots to free slots.
    remap: F,
    _key: PhantomData<Key>,
    _hx: PhantomData<Hx>,
}

impl<Key: KeyT, BF: BucketFn, F: MutPacked, Hx: Hasher<Key>> Default
    for PtrHash<Key, BF, F, Hx, Vec<u8>>
where
    PtrHashParams<BF>: Default,
{
    fn default() -> Self {
        PtrHash {
            params: PtrHashParams::default(),

            n: 0,
            parts: 0,
            shards: 0,
            parts_per_shard: 0,
            slots_total: 0,
            buckets_total: 0,
            slots: 0,
            lg_slots: 0,
            buckets: 0,
            rem_shards: FastReduce::new(0),
            rem_parts: FastReduce::new(0),
            rem_buckets: FastReduce::new(0),
            rem_buckets_total: FastReduce::new(0),
            rem_slots: MulReduce::new(1),
            seed: 0,
            pilots: vec![],
            remap: F::default(),
            _key: PhantomData,
            _hx: PhantomData,
        }
    }
}

/// Construction methods.
impl<Key: KeyT, BF: BucketFn, F: MutPacked, Hx: Hasher<Key>> PtrHash<Key, BF, F, Hx, Vec<u8>> {
    /// Create a new PtrHash instance from the given keys.
    ///
    /// NOTE: Only up to 2^40 keys are supported.
    ///
    /// Default parameters `alpha=0.98` and `c=9.0` should give fast
    /// construction that always succeeds, using `2.69 bits/key`.  Depending on
    /// the number of keys, you may be able to lower `c` (or slightly increase
    /// `alpha`) to reduce memory usage, at the cost of increasing construction
    /// time.
    ///
    /// By default, keys are partitioned into buckets of size ~250000, and parts are processed in parallel.
    /// This will use all available threads. To limit to fewer threads, use:
    /// ```rust
    /// let threads = 1;
    /// rayon::ThreadPoolBuilder::new()
    /// .num_threads(threads)
    /// .build_global()
    /// .unwrap();
    /// ```
    ///
    /// NOTE: Use `<PtrHash>::new()` or `DefaultPtrHash::new()` instead of simply `PtrHash::new()`.
    pub fn new(keys: &[Key], params: PtrHashParams<BF>) -> Self {
        let mut ptr_hash = Self::init(keys.len(), params);
        ptr_hash.compute_pilots(keys.par_iter()).unwrap();
        ptr_hash
    }

    /// Version that returns build statistics.
    #[doc(hidden)]
    pub fn new_with_stats(keys: &[Key], params: PtrHashParams<BF>) -> (Self, BucketStats) {
        let mut ptr_hash = Self::init(keys.len(), params);
        let stats = ptr_hash.compute_pilots(keys.par_iter()).unwrap();
        (ptr_hash, stats)
    }

    /// Fallible version of `new` that returns `None` if construction fails.
    pub fn try_new(keys: &[Key], params: PtrHashParams<BF>) -> Option<Self> {
        let mut ptr_hash = Self::init(keys.len(), params);
        ptr_hash.compute_pilots(keys.par_iter())?;
        Some(ptr_hash)
    }

    /// Same as `new` above, but takes a `ParallelIterator` over keys instead of a slice.
    /// The iterator must be cloneable for two reasons:
    /// - Construction can fail for the first seed (e.g. due to duplicate
    ///   hashes), in which case a new pass over keys is need.
    /// NOTE: The exact API may change here depending on what's most convenient to use.
    pub fn new_from_par_iter<'a>(
        n: usize,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
        params: PtrHashParams<BF>,
    ) -> Self {
        let mut ptr_hash = Self::init(n, params);
        ptr_hash.compute_pilots(keys);
        ptr_hash
    }

    /// PtrHash with random pilots, for benchmarking query speed.
    pub fn new_random(n: usize, params: PtrHashParams<BF>) -> Self {
        let mut ptr_hash = Self::init(n, params);
        let k = (0..ptr_hash.buckets_total)
            .map(|i| (i % 256) as Pilot)
            .collect();
        ptr_hash.pilots = MutPacked::new(k);
        let rem_s_total = FastReduce::new(ptr_hash.slots_total);
        let mut remap_vals = (ptr_hash.n..ptr_hash.slots_total)
            .map(|_| rem_s_total.reduce(random::<u64>()) as _)
            .collect_vec();
        remap_vals.radix_sort_unstable();
        ptr_hash.remap = MutPacked::new(remap_vals);
        ptr_hash.print_bits_per_element();
        ptr_hash
    }

    /// Only initialize the parameters; do not compute the pilots yet.
    fn init(n: usize, mut params: PtrHashParams<BF>) -> Self {
        assert!(n > 1, "Things break if n=1.");
        assert!(n < (1 << 40), "Number of keys must be less than 2^40.");
        let shards = match params.sharding {
            Sharding::None => 1,
            _ => n.div_ceil(params.keys_per_shard),
        };
        eprintln!("#shards: {}", shards);
        let keys_per_shard = n.div_ceil(shards);
        eprintln!("keys/shard: {}", keys_per_shard);

        // Compute the optimal number of slots per part.
        // - Smaller parts have better cache locality and hence faster construction.
        // - Larger parts have more uniform sizes, and hence fewer outliers with load factor close to 1.
        // We use the smallest power of 2 for which the probability that the
        // largest part has load factor <1 is large enough.
        let mut slots_per_part = params.slots_per_part.map_or(2, |s| s.next_power_of_two());
        assert!(
            slots_per_part <= u32::MAX as _,
            "Each part must have <2^32 slots"
        );

        let mut keys_per_part;
        let mut parts_per_shard;
        let mut buckets_per_part;

        let mut parts;
        let mut buckets_total;
        let mut slots_total;

        loop {
            keys_per_part = (params.alpha * slots_per_part as f64) as usize;
            parts_per_shard = keys_per_shard.div_ceil(keys_per_part);
            buckets_per_part = (keys_per_part as f64 / params.lambda).ceil() as usize;

            parts = shards * parts_per_shard;
            buckets_total = parts * buckets_per_part;
            slots_total = parts * slots_per_part;

            // Test if the probability of success is large enough.
            let exp_keys_per_part = n as f64 / parts as f64;
            let stddev = exp_keys_per_part.sqrt();
            // Expected size of largest part:
            // https://math.stackexchange.com/a/89147/91741:
            let stddevs_away = ((parts as f64).ln() * 2.).sqrt();
            let exp_max = exp_keys_per_part + stddev * stddevs_away;
            // Add a buffer of 1.5 stddev.
            let buf_max = exp_max + 1.5 * stddev;

            if buf_max < slots_per_part as f64 {
                eprintln!("Using slots per part: {slots_per_part}, expected keys {}, expected max keys: {} ({stddevs_away} σ)", exp_keys_per_part as usize, exp_max as usize);
                break;
            }

            // If slots_per_part was explicitly given, always use it.
            if params.slots_per_part.is_some() {
                eprintln!("Using user provided slots per part of {slots_per_part}, but it is likely too small for construction to succeed.");
                eprintln!(
                    "The largest part is expected to have around {} keys.",
                    exp_max as usize
                );
                break;
            }

            slots_per_part *= 2;
            assert!(
                slots_per_part <= u32::MAX as _,
                "Each part must have <2^32 slots"
            );
        }

        if params.print_stats {
            eprintln!("        keys: {n:>10}");
            eprintln!("      shards: {shards:>10}");
            eprintln!("       parts: {parts:>10}");
            eprintln!("   slots/prt: {slots_per_part:>10}");
            eprintln!("   slots tot: {slots_total:>10}");
            eprintln!("  real alpha: {:>10.4}", n as f64 / slots_total as f64);
            eprintln!(" buckets/prt: {buckets_per_part:>10}");
            eprintln!(" buckets tot: {buckets_total:>10}");
            eprintln!("keys/ bucket: {:>13.2}", n as f64 / buckets_total as f64);
        }
        params
            .bucket_fn
            .set_buckets_per_part(buckets_per_part as u64);

        Self {
            params,
            n,
            parts,
            shards,
            parts_per_shard,
            slots_total,
            slots: slots_per_part,
            lg_slots: slots_per_part.ilog2(),
            buckets_total,
            buckets: buckets_per_part,
            rem_shards: Rp::new(shards),
            rem_parts: Rp::new(parts),
            rem_buckets: Rb::new(buckets_per_part),
            rem_buckets_total: Rb::new(buckets_total),
            rem_slots: Rs::new(slots_per_part),
            seed: 0,
            pilots: Default::default(),
            remap: F::default(),
            _key: PhantomData,
            _hx: PhantomData,
        }
    }

    fn compute_pilots<'a>(
        &mut self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> Option<BucketStats> {
        let overall_start = std::time::Instant::now();
        // Initialize arrays;
        let mut taken: Vec<BitVec> = vec![];
        let mut pilots: Vec<u8> = vec![];

        let mut tries = 0;
        const MAX_TRIES: usize = 1;

        let mut rng = ChaCha8Rng::seed_from_u64(31415);

        // Loop over global seeds `s`.
        let stats = 's: loop {
            tries += 1;
            if tries > MAX_TRIES {
                eprintln!("Failed to find a global seed after {MAX_TRIES} tries.");
                return None;
            }
            if tries > 1 {
                eprintln!("Try {tries} for global seed.");
            }

            // Choose a global seed s.
            self.seed = rng.gen();

            // Reset output-memory.
            eprintln!("Pilots: {}MB", self.buckets_total / 1_000_000);
            pilots.clear();
            pilots.resize(self.buckets_total, 0);

            // TODO: Compress taken on the fly, instead of pre-allocating the entire thing.
            eprintln!("Taken: {}MB", self.parts * self.slots / 8 / 1_000_000);
            for taken in taken.iter_mut() {
                taken.clear();
                taken.resize(self.slots, false);
            }
            taken.resize_with(self.parts, || bitvec![0; self.slots]);

            // Iterate over shards.
            let shard_hashes = self.shards(keys.clone());
            let shard_pilots = pilots.chunks_mut(self.buckets * self.parts_per_shard);
            let shard_taken = taken.chunks_mut(self.parts_per_shard);
            let mut stats = BucketStats::default();
            // eprintln!("Num shards (keys) {}", shard_keys.());
            for (shard, (hashes, pilots, taken)) in
                izip!(shard_hashes, shard_pilots, shard_taken).enumerate()
            {
                // Determine the buckets.
                let start = std::time::Instant::now();
                let Some((hashes, part_starts)) = self.sort_parts(shard, hashes) else {
                    // Found duplicate hashes.
                    continue 's;
                };
                let start = log_duration("sort buckets", start);

                // Compute pilots.
                if let Some(shard_stats) =
                    self.build_shard(shard, &hashes, &part_starts, pilots, taken)
                {
                    stats.merge(shard_stats);
                    log_duration("find pilots", start);
                } else {
                    continue 's;
                }
            }

            // Found a suitable seed.
            if tries > 1 {
                eprintln!("Found seed after {tries} tries.");
            }

            break 's stats;
        };

        let start = std::time::Instant::now();
        self.remap_free_slots(taken);
        log_duration("remap free", start);

        // Pack the data.
        self.pilots = pilots;

        self.print_bits_per_element();
        log_duration("total build", overall_start);
        Some(stats)
    }

    fn remap_free_slots(&mut self, taken: Vec<BitVec>) {
        assert_eq!(
            taken.iter().map(|t| t.count_zeros()).sum::<usize>(),
            self.slots_total - self.n,
            "Not the right number of free slots left!\n total slots {} - n {}",
            self.slots_total,
            self.n
        );

        if !self.params.remap || self.slots_total == self.n {
            return;
        }

        // Compute the free spots.
        let mut v = Vec::with_capacity(self.slots_total - self.n);
        let get = |t: &Vec<BitVec>, idx: usize| t[idx / self.slots][idx % self.slots];
        for i in taken
            .iter()
            .enumerate()
            .flat_map(|(p, t)| {
                let offset = p * self.slots;
                t.iter_zeros().map(move |i| offset + i)
            })
            .take_while(|&i| i < self.n)
        {
            while !get(&taken, self.n + v.len()) {
                v.push(i as u64);
            }
            v.push(i as u64);
        }
        eprintln!("Remap len: {}", v.len());
        self.remap = MutPacked::new(v);
        eprintln!(
            "Remap size: {}MB = {}B",
            self.remap.size_in_bytes() / 1_000_000,
            self.remap.size_in_bytes()
        );
    }
}

/// Indexing methods.
impl<Key: KeyT, BF: BucketFn, F: Packed, Hx: Hasher<Key>, V: AsRef<[u8]>>
    PtrHash<Key, BF, F, Hx, V>
{
    /// Return the number of bits per element used for the pilots (`.0`) and the
    /// remapping (`.1)`.
    pub fn bits_per_element(&self) -> (f64, f64) {
        let pilots = self.pilots.as_ref().size_in_bytes() as f64 / self.n as f64;
        let remap = self.remap.size_in_bytes() as f64 / self.n as f64;
        (8. * pilots, 8. * remap)
    }

    pub fn print_bits_per_element(&self) {
        let (p, r) = self.bits_per_element();
        if self.params.print_stats {
            eprintln!(
                "bits/element: {:>13.2}  (pilots {p:4.2}, remap {r:4.2})",
                p + r
            );
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// index() always returns below this bound.
    pub fn max_index(&self) -> usize {
        self.slots_total
    }

    pub fn slots_per_part(&self) -> usize {
        self.slots
    }

    /// Get a non-minimal index of the given key.
    /// Use `index_minimal` to get a key in `[0, n)`.
    #[inline]
    pub fn index(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let pilot = self.pilots.as_ref().index(b);
        self.slot(hx, pilot)
    }

    /// Faster version of `index` for when there is only a single part.
    #[inline]
    pub fn index_single_part(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket_in_part(hx.high());
        let pilot = self.pilots.as_ref().index(b);
        self.slot_in_part(hx, pilot)
    }

    /// Get the index for `key` in `[0, n)`.
    ///
    /// Requires that the remap parameter is set to true.
    #[inline]
    pub fn index_minimal(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let p = self.pilots.as_ref().index(b);
        let slot = self.slot(hx, p);
        if slot < self.n {
            slot
        } else {
            self.remap.index(slot - self.n) as usize
        }
    }

    /// Takes an iterator over keys and returns an iterator over the indices of the keys.
    ///
    /// Uses a buffer of size K for prefetching ahead.
    // NOTE: It would be cool to use SIMD to determine buckets/positions in
    // parallel, but this is complicated, since SIMD doesn't support the
    // 64x64->128 multiplications needed in bucket/slot computations.
    #[inline]
    pub fn index_stream<'a, const B: usize, const MINIMAL: bool, Q: Borrow<Key> + 'a>(
        &'a self,
        keys: impl IntoIterator<Item = Q> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        let mut keys = keys.into_iter();

        // Ring buffers to cache the hash and bucket of upcoming queries.
        let mut next_hashes: [Hx::H; B] = [Hx::H::default(); B];
        let mut next_buckets: [usize; B] = [0; B];

        // Initialize and prefetch first B values.
        for idx in 0..B {
            let hx = self.hash_key(keys.next().unwrap().borrow());
            next_hashes[idx] = hx;

            next_buckets[idx] = self.bucket(next_hashes[idx]);
            crate::util::prefetch_index(self.pilots.as_ref(), next_buckets[idx]);
        }

        // Manual iterator implementation so we avoid the overhead and
        // non-inlining of Chain, and instead have a manual fold.
        struct It<
            'a,
            const B: usize,
            const MINIMAL: bool,
            Key: KeyT,
            Q: Borrow<Key> + 'a,
            KeyIt: Iterator<Item = Q> + 'a,
            BF: BucketFn,
            F: Packed,
            Hx: Hasher<Key>,
            V: AsRef<[u8]>,
        > {
            ph: &'a PtrHash<Key, BF, F, Hx, V>,
            keys: KeyIt,
            next_hashes: [Hx::H; B],
            next_buckets: [usize; B],
        }

        impl<
                'a,
                const B: usize,
                const MINIMAL: bool,
                Key: KeyT,
                Q: Borrow<Key> + 'a,
                KeyIt: Iterator<Item = Q> + 'a,
                BF: BucketFn,
                F: Packed,
                Hx: Hasher<Key>,
                V: AsRef<[u8]>,
            > Iterator for It<'a, B, MINIMAL, Key, Q, KeyIt, BF, F, Hx, V>
        {
            type Item = usize;
            #[inline(always)]
            fn next(&mut self) -> Option<usize> {
                todo!();
            }

            #[inline(always)]
            fn fold<BB, FF>(mut self, init: BB, mut f: FF) -> BB
            where
                Self: Sized,
                FF: FnMut(BB, Self::Item) -> BB,
            {
                let mut accum = init;
                let mut i = 0;

                for key in self.keys {
                    let next_hash = self.ph.hash_key(key.borrow());
                    let idx = i % B;
                    let cur_hash = self.next_hashes[idx];
                    let cur_bucket = self.next_buckets[idx];
                    self.next_hashes[idx] = next_hash;
                    self.next_buckets[idx] = self.ph.bucket(self.next_hashes[idx]);
                    crate::util::prefetch_index(self.ph.pilots.as_ref(), self.next_buckets[idx]);
                    let pilot = self.ph.pilots.as_ref().index(cur_bucket);
                    let slot = self.ph.slot(cur_hash, pilot);

                    let slot = if MINIMAL && slot >= self.ph.n {
                        self.ph.remap.index(slot - self.ph.n) as usize
                    } else {
                        slot
                    };

                    accum = f(accum, slot);
                    i += 1;
                }

                for _ in 0..B {
                    let idx = i % B;
                    let cur_hash = self.next_hashes[idx];
                    let cur_bucket = self.next_buckets[idx];
                    let pilot = self.ph.pilots.as_ref().index(cur_bucket);
                    let slot = self.ph.slot(cur_hash, pilot);

                    let slot = if MINIMAL && slot >= self.ph.n {
                        self.ph.remap.index(slot - self.ph.n) as usize
                    } else {
                        slot
                    };

                    accum = f(accum, slot);
                    i += 1;
                }

                accum
            }
        }
        It::<B, MINIMAL, _, _, _, _, _, _, _> {
            ph: self,
            keys,
            next_hashes,
            next_buckets,
        }
    }

    /// Takes an iterator over keys and returns an iterator over the indices of the keys.
    ///
    /// Queries in batches of size K.
    ///
    /// NOTE: Does not process the remainder
    #[cfg(feature = "unstable")]
    #[inline]
    pub fn index_batch_exact<'a, const K: usize, const MINIMAL: bool>(
        &'a self,
        xs: impl IntoIterator<Item = &'a Key> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        let mut buckets: [usize; K] = [0; K];

        // Work on chunks of size K.
        let mut f = {
            #[inline(always)]
            move |hx: [Hx::H; K]| {
                // Prefetch.
                for idx in 0..K {
                    buckets[idx] = self.bucket(hx[idx]);
                    crate::util::prefetch_index(self.pilots.as_ref(), buckets[idx]);
                }
                // Query.
                (0..K).map(
                    #[inline(always)]
                    move |idx| {
                        let pilot = self.pilots.as_ref().index(buckets[idx]);
                        let slot = self.slot(hx[idx], pilot);
                        if MINIMAL && slot >= self.n {
                            self.remap.index(slot - self.n) as usize
                        } else {
                            slot
                        }
                    },
                )
            }
        };
        let array_chunks = xs.into_iter().map(|x| self.hash_key(x)).array_chunks::<K>();
        array_chunks.into_iter().flat_map(
            #[inline(always)]
            move |chunk| f(chunk),
        )
        // .chain(f(&array_chunks
        //     .into_remainder()
        //     .unwrap_or_default()
        //     .into_iter()))
    }

    /// A variant of index_batch_exact that scales better with K.
    /// Somehow the version above has pretty constant speed regardless of K.
    #[inline]
    pub fn index_batch_exact2<'a, const K: usize, const MINIMAL: bool>(
        &'a self,
        xs: impl IntoIterator<Item = &'a Key, IntoIter: ExactSizeIterator> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        let mut buckets: [usize; K] = [0; K];
        let mut hs: [Hx::H; K] = [Hx::H::default(); K];

        let mut xs = xs
            .into_iter()
            .map(|x| self.hash_key(x))
            .chain([Default::default(); K]);
        for i in 0..K {
            hs[i] = xs.next().unwrap();
        }
        let mut idx = K;
        xs.map(move |hx| {
            if idx == K {
                idx = 0;
                // Prefetch.
                for idx in 0..K {
                    buckets[idx] = self.bucket(hs[idx]);
                    crate::util::prefetch_index(self.pilots.as_ref(), buckets[idx]);
                }
            }

            // Query.
            let pilot = self.pilots.as_ref().index(buckets[idx]);
            let slot = self.slot(hs[idx], pilot);

            // Update hash in current pos and increment.
            hs[idx] = hx;
            idx += 1;

            // Remap?
            if MINIMAL && slot >= self.n {
                self.remap.index(slot - self.n) as usize
            } else {
                slot
            }
        })
    }

    fn hash_key(&self, x: &Key) -> Hx::H {
        Hx::hash(x, self.seed)
    }

    fn hash_pilot(&self, p: Pilot) -> PilotHash {
        MulHash::hash(&p, self.seed)
    }

    fn shard(&self, hx: Hx::H) -> usize {
        self.rem_shards.reduce(hx.high())
    }

    fn part(&self, hx: Hx::H) -> usize {
        self.rem_parts.reduce(hx.high())
    }

    /// Map `hx_remainder` to a bucket in the range [0, self.b).
    /// Hashes <self.p1 are mapped to large buckets [0, self.p2).
    /// Hashes >=self.p1 are mapped to small [self.p2, self.b).
    ///
    /// (Unless SPLIT_BUCKETS is false, in which case all hashes are mapped to [0, self.b).)
    fn bucket_in_part(&self, x: u64) -> usize {
        if BF::B_OUTPUT {
            self.params.bucket_fn.call(x) as usize
        } else {
            self.rem_buckets.reduce(self.params.bucket_fn.call(x))
        }
    }

    /// See bucket.rs for additional implementations.
    /// Returns the offset in the slots array for the current part and the bucket index.
    fn bucket(&self, hx: Hx::H) -> usize {
        if BF::LINEAR {
            return self.rem_buckets_total.reduce(hx.high());
        }

        // Extract the high bits for part selection; do normal bucket
        // computation within the part using the remaining bits.
        // NOTE: This is somewhat slow, but doing better is hard.
        let (part, hx) = self.rem_parts.reduce_with_remainder(hx.high());
        let bucket = self.bucket_in_part(hx);
        part * self.buckets + bucket
    }

    /// Slot uses the 64 low bits of the hash.
    fn slot(&self, hx: Hx::H, pilot: u64) -> usize {
        (self.part(hx) << self.lg_slots) + self.slot_in_part(hx, pilot)
    }

    fn slot_in_part(&self, hx: Hx::H, pilot: Pilot) -> usize {
        self.slot_in_part_hp(hx, self.hash_pilot(pilot))
    }

    /// Slot uses the 64 low bits of the hash.
    fn slot_in_part_hp(&self, hx: Hx::H, hp: PilotHash) -> usize {
        // NOTE: Fastmod s is slower since it needs two multiplications instead of 1.
        // NOTE: A simple &(s-1) mask is not sufficient, since it only uses the low order bits.
        //       The part() and bucket() functions only use high order bits, which
        //       would leave the middle bits unused, causing hash collisions.
        self.rem_slots.reduce(hx.low() ^ hp)
    }
}
