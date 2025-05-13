#![cfg_attr(feature = "unstable", feature(iter_array_chunks))]
//! # PtrHash: Minimal Perfect Hashing at RAM Throughput
//!
//! PtrHash builds a _minimal perfect hash function_, that is,
//! a hash function that maps a fixed set of keys to `{0, ..., n-1}`.
//!
//! PtrHash was developed for large key sets of at least 1 million keys, and has been tested up to 10^11 keys.
//! It only uses 2.4 bits per key.
//!
//! It can also be used for arbitrary small sets.
//! In this case, the space efficiency will be less due to a relatively large constant overhead.
//!
//! See the GitHub [readme](https://github.com/ragnargrootkoerkamp/ptrhash)
//! or paper ([arXiv](https://arxiv.org/abs/2502.15539), [blog version](https://curiouscoding.nl/posts/ptrhash/))
//! for details on the algorithm and performance.
//!
//! Usage example:
//! ```rust
//! use ptr_hash::{PtrHash, PtrHashParams};
//!
//! // Generate some random keys.
//! let n = 1_000_000;
//! let keys = ptr_hash::util::generate_keys(n);
//!
//! // Build the datastructure.
//! // NOTE: For small sets, say <1M keys, use `PtrHashParams::default_fast()` instead.
//! // The default parameters are optimized for large sets, and need large (>100k or so) inputs
//! // to ensure internal bucket sizes don't deviate too much from their expectation.
//! let mphf = <PtrHash>::new(&keys, PtrHashParams::default());
//!
//! // Get the index of a key.
//! let key = 0;
//! let idx = mphf.index(&key);
//! assert!(idx < n);
//!
//! // Get the non-minimal index of a key.
//! // Can be slightly faster returns keys up to `n/alpha ~ 1.01*n`.
//! let idx = mphf.index_no_remap(&key);
//! // `max_index` returns an upper bound on the non-remapped index.
//! assert!(idx < mphf.max_index());
//!
//! // An iterator over the indices of the keys.
//! // 32: number of iterations ahead to prefetch.
//! // true: remap to a minimal key in [0, n).
//! // _: placeholder to infer the type of keys being iterated.
//! let indices = mphf.index_stream::<32, true, _>(&keys);
//! assert_eq!(indices.sum::<usize>(), (n * (n - 1)) / 2);
//!
//! // Query a batch of keys.
//! let keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
//! let mut indices = mphf.index_batch::<16, true, _>(keys);
//! indices.sort();
//! for i in 0..indices.len()-1 {
//!     assert!(indices[i] != indices[i+1]);
//! }
//!
//! // Test that all items map to different indices
//! let mut taken = vec![false; n];
//! for key in keys {
//!     let idx = mphf.index(&key);
//!     assert!(!taken[idx]);
//!     taken[idx] = true;
//! }
//! ```
//!
//! ## Partitioning
//!
//! By default, PtrHash partitions the keys into multiple parts.
//! This speeds up construction in two ways:
//! - smaller parts have better cache locality, and
//! - parts can be constructed in parallel.
//!
//! However, at query time there is a small overhead to compute the part of each key.
//! To achieve slightly faster queries, set [`PtrHashParams::single_part`] to `true`,
//! and then use [`PtrHash::index_single_part()`] instead of [`PtrHash::index()`].

/// Customizable Hasher trait.
pub mod hash;
/// Extendable backing storage trait and types.
pub mod pack;
/// Some internal logging and testing utilities.
pub mod util;

pub mod bucket_fn;
mod bucket_idx;
mod build;
mod fastmod;
mod reduce;
mod shard;
mod sort_buckets;
#[doc(hidden)]
pub mod stats;
#[cfg(test)]
mod test;

use bitvec::{bitvec, vec::BitVec};
use bucket_fn::BucketFn;
use bucket_fn::CubicEps;
use bucket_fn::Linear;
use bucket_fn::SquareEps;
use cacheline_ef::CachelineEfVec;
use fastmod::FM32;
use itertools::izip;
use itertools::Itertools;
use log::trace;
use mem_dbg::MemSize;
use pack::MutPacked;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
pub use shard::Sharding;
use stats::BucketStats;
use std::array::from_fn;
use std::{borrow::Borrow, default::Default, marker::PhantomData, time::Instant};

use crate::{hash::*, pack::Packed, reduce::*, util::log_duration};

/// Parameters for PtrHash construction.
///
/// While all fields are public, prefer one of the default functions,
/// [`PtrHashParams::default()`], [`PtrHashParams::default_fast()`], or
/// [`PtrHashParams::default_compact()`].
#[derive(Clone, Copy, Debug, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", deep_copy)]
pub struct PtrHashParams<BF> {
    /// Set to false to disable remapping to a minimal PHF.
    pub remap: bool,
    /// Use `n/alpha` slots approximately.
    pub alpha: f64,
    /// Use average bucket size lambda.
    pub lambda: f64,
    /// Bucket function
    pub bucket_fn: BF,
    /// Upper bound on number of keys per shard.
    /// Default is 2^32, or 32GB of hashes per shard.
    pub keys_per_shard: usize,
    /// When true, write each shard to a file instead of iterating multiple
    /// times.
    pub sharding: Sharding,

    /// Force using a single part, so that [`PtrHash::index_single_part()`] can be used.
    ///
    /// This makes construction up to 5x slower, but can make queries up to 30% faster.
    pub single_part: bool,
}

impl PtrHashParams<Linear> {
    /// Parameters for faster construction and querying, and for inputs <1M.
    ///
    /// Takes `3.0` bits/key, and can be up to 2x faster to query than the default version.
    /// - `alpha=0.99`
    /// - `lambda=3.0`
    /// - `bucket_fn=Linear`
    pub fn default_fast() -> Self {
        Self {
            remap: true,
            alpha: 0.99,
            lambda: 3.0,
            bucket_fn: Linear,
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
            single_part: false,
        }
    }
}

#[doc(hidden)]
impl PtrHashParams<SquareEps> {
    pub fn default_square() -> Self {
        Self {
            remap: true,
            alpha: 0.99,
            lambda: 3.5,
            bucket_fn: SquareEps,
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
            single_part: false,
        }
    }
}

impl PtrHashParams<CubicEps> {
    /// Default parameters for inputs >1M keys.
    ///
    /// Takes `2.4` bits/key, and trades off space and speed.
    /// - `alpha=0.99`
    /// - `lambda=3.5`
    /// - `bucket_fn=CubicEps`
    pub fn default() -> Self {
        Self {
            remap: true,
            alpha: 0.99,
            lambda: 3.5,
            bucket_fn: CubicEps,
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
            single_part: false,
        }
    }

    /// Default 'compact' parameters.
    ///
    /// Takes `2.1` bits/key, but is typically 2x slower to construct than the default version.
    /// This occasionally fails construction. If so, try again with decreased `lambda`.
    /// - `alpha=0.99`
    /// - `lambda=4.0`
    /// - `bucket_fn=CubicEps`
    pub fn default_compact() -> Self {
        Self {
            remap: true,
            alpha: 0.99,
            lambda: 3.9,
            bucket_fn: CubicEps,
            keys_per_shard: 1 << 31,
            sharding: Sharding::None,
            single_part: false,
        }
    }
}

impl Default for PtrHashParams<CubicEps> {
    fn default() -> Self {
        Self::default()
    }
}

/// Type alias to simplify construction.
///
/// [`PtrHash`] has a large number of generics, partly to support epserde.
/// [`DefaultPtrHash`] fills in most values.
///
/// Use this as [`DefaultPtrHash::new()`] or `<DefaultPtrHash>::new()`.
pub type DefaultPtrHash<Hx = hash::IntHash, Key = u64, BF = bucket_fn::CubicEps> =
    PtrHash<Key, BF, CachelineEfVec, Hx, Vec<u8>>;

/// Trait that keys must satisfy.
pub trait KeyT: Send + Sync + std::hash::Hash {}
impl<T: Send + Sync + std::hash::Hash> KeyT for T {}

// Some fixed algorithmic decisions.
type Rp = FastReduce;
type Rb = FastReduce;
type RemSlots = FM32;
type Pilot = u64;
type PilotHash = u64;

/// PtrHash datastructure.
/// It is recommended to use PtrHash with default types.
///
/// - `Key`: The type of keys to hash.
/// - `BF`: The bucket function to use. Inferred from `PtrHashParams` when calling `PtrHash::new()`.
/// - `F`: The packing to use for remapping free slots, default `CachelineEf`.
/// - `Hx`: The hasher to use for keys, default `FxHash` for integers, but consider
///       `hash::StringHash` (using `gxhash`) for strings, or `hash::StringHash128` when the number of string keys is very
///       large.
/// - `V`: The pilots type. Usually `Vec<u8>`, or `&[u8]` for Epserde.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone, MemSize)]
pub struct PtrHash<
    Key: KeyT + ?Sized = u64,
    BF: BucketFn = bucket_fn::CubicEps,
    F: Packed = CachelineEfVec,
    Hx: KeyHasher<Key> = hash::IntHash,
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
    /// The number of buckets per part.
    buckets: usize,

    // Precomputed fast modulo operations.
    /// Fast %shards.
    rem_shards: Rp,
    /// Fast %parts.
    rem_parts: Rp,
    /// Fast %b.
    rem_buckets: Rb,
    /// Fast %b_total.
    rem_buckets_total: Rb,

    /// Fast %s when there is only a single part.
    rem_slots: RemSlots,

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

/// An empty PtrHash instance. Mostly useless, but may be convenient.
impl<Key: KeyT, BF: BucketFn, F: MutPacked, Hx: KeyHasher<Key>> Default
    for PtrHash<Key, BF, F, Hx, Vec<u8>>
where
    PtrHashParams<BF>: Default,
{
    fn default() -> Self {
        PtrHash {
            params: <PtrHashParams<BF> as Default>::default(),

            n: 0,
            parts: 0,
            shards: 0,
            parts_per_shard: 0,
            slots_total: 0,
            buckets_total: 0,
            slots: 0,
            buckets: 0,
            rem_shards: FastReduce::new(0),
            rem_parts: FastReduce::new(0),
            rem_buckets: FastReduce::new(0),
            rem_buckets_total: FastReduce::new(0),
            rem_slots: RemSlots::new(0),
            seed: 0,
            pilots: vec![],
            remap: F::default(),
            _key: PhantomData,
            _hx: PhantomData,
        }
    }
}

/// Construction methods.
impl<Key: KeyT, BF: BucketFn, F: MutPacked, Hx: KeyHasher<Key>> PtrHash<Key, BF, F, Hx, Vec<u8>> {
    /// Create a new PtrHash instance from the given keys.
    ///
    /// Use `<PtrHash>::new()` or `DefaultPtrHash::new()` instead of simply `PtrHash::new()` to
    /// get the default values for the generics.
    ///
    /// NOTE: This panics when construction fails after 10 attempts.
    /// This should be rare, but can happen if we are unlucky with the rng seeds.
    /// Consider calling [`PtrHash::try_new()`] instead.
    ///
    /// NOTE: Only up to 2^40 keys are supported.
    pub fn new(keys: &[Key], params: PtrHashParams<BF>) -> Self {
        let mut ptr_hash = Self::init(keys.len(), params);
        ptr_hash
            .compute_pilots(keys.par_iter())
            .expect("Unable to construct PtrHash after 10 tries. Try using a better hash or decreasing lambda.");
        ptr_hash
    }

    /// Version that returns build statistics.
    #[doc(hidden)]
    pub fn new_with_stats(keys: &[Key], params: PtrHashParams<BF>) -> (Self, BucketStats) {
        let mut ptr_hash = Self::init(keys.len(), params);
        let stats = ptr_hash
            .compute_pilots(keys.par_iter())
            .expect("Unable to construct PtrHash after 10 tries. Try using a better hash or decreasing lambda.");
        (ptr_hash, stats)
    }

    /// Fallible version of `new` that returns `None` if construction fails.
    /// This can happen when `lambda` is too larger (e.g. for `default_compact`
    /// parameters) and the eviction chains become too long.
    pub fn try_new(keys: &[Key], params: PtrHashParams<BF>) -> Option<Self> {
        let mut ptr_hash = Self::init(keys.len(), params);
        ptr_hash.compute_pilots(keys.par_iter())?;
        Some(ptr_hash)
    }

    /// Same as `new` above, but takes a `ParallelIterator` over keys instead of a slice.
    ///
    /// The iterator must be cloneable, since construction can fail for the
    /// first seed (e.g. due to duplicate hashes), in which case a new pass over
    /// keys is need.
    pub fn new_from_par_iter<'a>(
        n: usize,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
        params: PtrHashParams<BF>,
    ) -> Self {
        let mut ptr_hash = Self::init(n, params);
        ptr_hash.compute_pilots(keys);
        ptr_hash
    }

    /// Only initialize the parameters; do not compute the pilots yet.
    fn init(n: usize, mut params: PtrHashParams<BF>) -> Self {
        assert!(n < (1 << 40), "Number of keys must be less than 2^40.");

        let shards = match params.sharding {
            Sharding::None => 1,
            _ => n.div_ceil(params.keys_per_shard),
        };

        // Formula of Vigna, eps-cost-sharding: https://arxiv.org/abs/2503.18397
        // (1-alpha)/2, so that on average we still have some room to play with.
        let eps = (1.0 - params.alpha) / 2.0;
        let x = n as f64 * eps * eps / 2.0;
        let target_parts = x / x.ln();
        let parts_per_shard = (target_parts.floor() as usize) / shards;
        let parts = parts_per_shard.max(1) * shards;

        let keys_per_part = n / parts;
        let parts_per_shard = parts / shards;
        let mut slots_per_part = (keys_per_part as f64 / params.alpha) as usize;
        // Avoid powers of two, since then %S does not depend on all bits.
        if slots_per_part.is_power_of_two() {
            slots_per_part += 1;
        }
        let slots_total = parts * slots_per_part;
        // Add a few extra buckets to avoid collisions for small n.
        let buckets_per_part = (keys_per_part as f64 / params.lambda).ceil() as usize + 3;
        let buckets_total = parts * buckets_per_part;

        trace!("        keys: {n:>10}");
        trace!("      shards: {shards:>10}");
        trace!("       parts: {parts:>10}");
        trace!("   slots/prt: {slots_per_part:>10}");
        trace!("   slots tot: {slots_total:>10}");
        trace!("  real alpha: {:>10.4}", n as f64 / slots_total as f64);
        trace!(" buckets/prt: {buckets_per_part:>10}");
        trace!(" buckets tot: {buckets_total:>10}");
        trace!("keys/ bucket: {:>13.2}", n as f64 / buckets_total as f64);

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
            buckets_total,
            buckets: buckets_per_part,
            rem_shards: Rp::new(shards),
            rem_parts: Rp::new(parts),
            rem_buckets: Rb::new(buckets_per_part),
            rem_buckets_total: Rb::new(buckets_total),
            rem_slots: RemSlots::new(slots_per_part.max(1)), // fix for n=0
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
        const MAX_TRIES: usize = 10;

        let mut rng = ChaCha8Rng::seed_from_u64(31415);

        // Loop over global seeds `s`.
        let stats = 's: loop {
            tries += 1;
            if tries > MAX_TRIES {
                log::error!("PtrHash failed to find a global seed after {MAX_TRIES} tries.");
                return None;
            }
            if tries > 1 {
                log::warn!("NEW TRY Try {tries} for global seed.");
            }

            // Choose a global seed s.
            self.seed = rng.random();

            // Reset output-memory.
            pilots.clear();
            pilots.resize(self.buckets_total, 0);

            // TODO: Compress taken on the fly, instead of pre-allocating the entire thing.
            for taken in taken.iter_mut() {
                taken.clear();
                taken.resize(self.slots, false);
            }
            taken.resize_with(self.parts, || bitvec![0; self.slots]);

            // Iterate over shards.
            let shard_hashes = self.shards(keys.clone());
            // Avoid chunks_mut(0) when n=0.
            let shard_pilots = pilots.chunks_mut((self.buckets * self.parts_per_shard).max(1));
            let shard_taken = taken.chunks_mut(self.parts_per_shard);
            let mut stats = BucketStats::default();
            // eprintln!("Num shards (keys) {}", shard_keys.());
            for (shard, (hashes, pilots, taken)) in
                izip!(shard_hashes, shard_pilots, shard_taken).enumerate()
            {
                // Determine the buckets.
                let start = std::time::Instant::now();
                let Some((hashes, part_starts)) = self.sort_parts(shard, hashes) else {
                    trace!("Found duplicate hashes");
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
                    trace!("Could not find pilots");
                    continue 's;
                }
            }

            let start = std::time::Instant::now();
            let remap = self.remap_free_slots(&taken);
            log_duration("remap free", start);
            if remap.is_err() {
                trace!("Failed to construct CachelineEF");
                continue 's;
            }

            break 's stats;
        };

        // Pack the data.
        self.pilots = pilots;

        let (p, r) = self.bits_per_element();
        trace!("bits/element: {}", p + r);
        log_duration("total build", overall_start);
        Some(stats)
    }

    fn remap_free_slots(&mut self, taken: &Vec<BitVec>) -> Result<(), ()> {
        assert_eq!(
            taken.iter().map(|t| t.count_zeros()).sum::<usize>(),
            self.slots_total - self.n,
            "Not the right number of free slots left!\n total slots {} - n {}",
            self.slots_total,
            self.n
        );

        if !self.params.remap || self.slots_total == self.n {
            return Ok(());
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
        self.remap = MutPacked::try_new(v).ok_or(())?;
        Ok(())
    }
}

/// Indexing methods.
impl<Key: KeyT, BF: BucketFn, F: Packed, Hx: KeyHasher<Key>, V: AsRef<[u8]>>
    PtrHash<Key, BF, F, Hx, V>
{
    /// Return the number of bits per element used for the pilots (`.0`) and the
    /// remapping (`.1`).
    pub fn bits_per_element(&self) -> (f64, f64) {
        let pilots = self.pilots.as_ref().size_in_bytes() as f64 / self.n as f64;
        let remap = self.remap.size_in_bytes() as f64 / self.n as f64;
        (8. * pilots, 8. * remap)
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// `self.index_no_remap()` always returns below this bound.
    /// Should be around `n/alpha ~ 1.01*n`.
    pub fn max_index(&self) -> usize {
        self.slots_total
    }

    pub fn slots_per_part(&self) -> usize {
        self.slots
    }

    /// Get a non-minimal index of the given key, in `[0, n/alpha)`.
    /// Use `index` to get a key in `[0, n)`.
    #[inline]
    pub fn index_no_remap(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let pilot = self.pilots.as_ref().index(b);
        self.slot(hx, pilot)
    }

    /// Faster version of `index` for when there is only a single part.
    /// Use only when there is indeed a single part, i.e., after constructing
    /// with [`PtrHashParams::single_part`] set to `true`.
    #[inline]
    pub fn index_single_part(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket_in_part(hx.high());
        let pilot = self.pilots.as_ref().index(b);
        let slot = self.slot_in_part(hx, pilot);
        if slot < self.n {
            slot
        } else {
            self.remap.index(slot - self.n) as usize
        }
    }

    /// Get the index for `key` in `[0, n)`.
    #[inline]
    pub fn index(&self, key: &Key) -> usize {
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
    /// Uses a buffer of size `B` for prefetching ahead. `B=32` should be a good choice.
    /// By default, set `MINIMAL` to false when you do not need remapp
    /// The iterator can return either `Q=Key` or `Q=&Key`.
    ///
    /// See the module-level documentation for an example.
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
        let mut leftover = B;
        for idx in 0..B {
            let hx = keys
                .next()
                .map(|k| {
                    leftover -= 1;
                    self.hash_key(k.borrow())
                })
                .unwrap_or_default();
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
            Hx: KeyHasher<Key>,
            V: AsRef<[u8]>,
        > {
            ph: &'a PtrHash<Key, BF, F, Hx, V>,
            keys: KeyIt,
            next_hashes: [Hx::H; B],
            next_buckets: [usize; B],
            leftover: usize,
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
                Hx: KeyHasher<Key>,
                V: AsRef<[u8]>,
            > Iterator for It<'a, B, MINIMAL, Key, Q, KeyIt, BF, F, Hx, V>
        {
            type Item = usize;
            fn next(&mut self) -> Option<usize> {
                unimplemented!("Use a method that calls `fold()` instead.");
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

                for _ in 0..B - self.leftover {
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
            leftover,
        }
    }

    /// Query a batch of `K` keys at once.
    ///
    /// Input can be either `[Key; K]` or `[&Key; K]`.
    #[inline]
    pub fn index_batch<'a, const K: usize, const MINIMAL: bool, Q: Borrow<Key> + 'a>(
        &'a self,
        xs: [Q; K],
    ) -> [usize; K] {
        let hashes = xs.map(|x| self.hash_key(x.borrow()));
        let mut buckets: [usize; K] = [0; K];

        // Prefetch.
        for idx in 0..K {
            buckets[idx] = self.bucket(hashes[idx]);
            crate::util::prefetch_index(self.pilots.as_ref(), buckets[idx]);
        }
        // Query.
        from_fn(
            #[inline(always)]
            move |idx| {
                let pilot = self.pilots.as_ref().index(buckets[idx]);
                let slot = self.slot(hashes[idx], pilot);
                if MINIMAL && slot >= self.n {
                    self.remap.index(slot - self.n) as usize
                } else {
                    slot
                }
            },
        )
    }

    /// Takes an iterator over keys and returns an iterator over the indices of the keys.
    ///
    /// Queries in batches of size K.
    ///
    /// NOTE: Does not process the remainder
    #[doc(hidden)]
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
    #[doc(hidden)]
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
        hash::C.wrapping_mul(p ^ self.seed)
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
        (self.part(hx) * self.slots) + self.slot_in_part(hx, pilot)
    }

    fn slot_in_part(&self, hx: Hx::H, pilot: Pilot) -> usize {
        self.slot_in_part_hp(hx, self.hash_pilot(pilot))
    }

    /// Slot uses the 64 low bits of the hash.
    fn slot_in_part_hp(&self, hx: Hx::H, hp: PilotHash) -> usize {
        self.rem_slots.reduce(hx.low() ^ hp)
    }
}
