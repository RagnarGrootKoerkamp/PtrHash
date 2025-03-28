//! Implementations of the `Hash` trait that abstracts over 64 and 128-bit hashes.
use mem_dbg::MemSize;

use crate::KeyT;
use std::fmt::Debug;

/// The `Hasher` trait returns a 64 or 128-bit `Hash`. From this, two `u64` values are extracted.
///
/// When 64-bit hashes are enough, we simply return the same hash (the `u64`
/// `Self` value) as the low and high part.
///
/// When 128-bit hashes are needed, the two functions return the low/high half of bits.
///
/// Our method never needs the full hash value, and instead uses the two hashes
/// in different places to extract sufficient entropy.
pub trait Hash: Copy + Debug + Default + Send + Sync + Eq + rdst::RadixKey {
    /// Returns the low 64bits of the hash.
    fn low(&self) -> u64;
    /// Returns the high 64bits of the hash.
    fn high(&self) -> u64;
}

impl Hash for u64 {
    fn low(&self) -> u64 {
        *self
    }
    fn high(&self) -> u64 {
        *self
    }
}

impl Hash for u128 {
    fn low(&self) -> u64 {
        *self as u64
    }
    fn high(&self) -> u64 {
        (*self >> 64) as u64
    }
}

/// Wrapper trait for various hash functions.
pub trait Hasher<Key: ?Sized>: Clone + Sync {
    type H: Hash;
    fn hash(x: &Key, seed: u64) -> Self::H;
}

fn to_bytes<Key: ?Sized>(x: &Key) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x as *const Key as *const u8, std::mem::size_of_val(x)) }
}

// A. u64-only hashers
/// Multiply the key by a mixing constant.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct MulHash;
/// Pass the key through unchanged.
/// Used for benchmarking.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct NoHash;

// B. Fast hashers that are always included.
/// Good for hashing `u64` and smaller keys.
/// Note that this doesn't use a seed, so while it is a bijection on `u64` keys,
/// larger keys will give unfixable collisions.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone, MemSize)]
pub struct FxHash;
/// Default hash function for strings.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone, MemSize)]
pub struct Xx64;
/// Fast good 128bit hash, when hashing >>10^9 keys.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Xx128;

/// Very fast weak 64bit hash with more quality than FxHash.
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Murmur2_64;
/// Fast weak 128bit hash for integers.
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct FastMurmur3_128;

// C. Additional higher quality but slower hashers.
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Murmur3_128;
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Highway64;
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Highway128;
/// Fast good 64bit hash.
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct City64;
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct City128;
/// Fast good 64bit hash.
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Wy64;

#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Metro64;
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Metro128;
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Spooky64;
#[cfg(feature = "hashers")]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[derive(Clone)]
pub struct Spooky128;

// Hash implementations.

// A. u64-only hashers.
impl MulHash {
    // Reuse the mixing constant from MurmurHash.
    // pub const C: u64 = 0xc6a4a7935bd1e995;
    // Reuse the mixing constant from FxHash.
    pub const C: u64 = 0x517cc1b727220a95;
}
impl Hasher<u64> for MulHash {
    type H = u64;
    fn hash(x: &u64, _seed: u64) -> u64 {
        Self::C.wrapping_mul(*x)
    }
}
impl Hasher<u64> for NoHash {
    type H = u64;
    fn hash(x: &u64, _seed: u64) -> u64 {
        *x
    }
}

// B. Fast hashers that are always included.
impl<Key: KeyT> Hasher<Key> for FxHash {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        fxhash::hash64(x)
    }
}

// XX64

impl Hasher<u64> for Xx64 {
    type H = u64;
    #[inline(always)]
    fn hash(x: &u64, seed: u64) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(to_bytes(x), seed)
    }
}
impl Hasher<Box<u64>> for Xx64 {
    type H = u64;
    #[inline(always)]
    fn hash(x: &Box<u64>, seed: u64) -> u64 {
        let x = **x;
        xxhash_rust::xxh3::xxh3_64_with_seed(to_bytes(&x), seed)
    }
}
impl Hasher<[u8]> for Xx64 {
    type H = u64;
    #[inline(always)]
    fn hash(x: &[u8], seed: u64) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(to_bytes(x), seed)
    }
}
impl<const N: usize> Hasher<[u8; N]> for Xx64 {
    type H = u64;
    #[inline(always)]
    fn hash(x: &[u8; N], seed: u64) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(to_bytes(x), seed)
    }
}
impl Hasher<&[u8]> for Xx64 {
    type H = u64;
    #[inline(always)]
    fn hash(x: &&[u8], seed: u64) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(to_bytes(*x), seed)
    }
}
impl<const N: usize> Hasher<&[u8; N]> for Xx64 {
    type H = u64;
    #[inline(always)]
    fn hash(x: &&[u8; N], seed: u64) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(to_bytes(x), seed)
    }
}
impl Hasher<Vec<u8>> for Xx64 {
    type H = u64;
    #[inline(always)]
    fn hash(x: &Vec<u8>, seed: u64) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(to_bytes(x.as_slice()), seed)
    }
}

// XX128

impl Hasher<u64> for Xx128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &u64, seed: u64) -> u128 {
        xxhash_rust::xxh3::xxh3_128_with_seed(to_bytes(x), seed)
    }
}
impl Hasher<Box<u64>> for Xx128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &Box<u64>, seed: u64) -> u128 {
        let x = **x;
        xxhash_rust::xxh3::xxh3_128_with_seed(to_bytes(&x), seed)
    }
}
impl Hasher<[u8]> for Xx128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &[u8], seed: u64) -> u128 {
        xxhash_rust::xxh3::xxh3_128_with_seed(to_bytes(x), seed)
    }
}
impl<const N: usize> Hasher<[u8; N]> for Xx128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &[u8; N], seed: u64) -> u128 {
        xxhash_rust::xxh3::xxh3_128_with_seed(to_bytes(x), seed)
    }
}
impl Hasher<&[u8]> for Xx128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &&[u8], seed: u64) -> u128 {
        xxhash_rust::xxh3::xxh3_128_with_seed(to_bytes(*x), seed)
    }
}
impl<const N: usize> Hasher<&[u8; N]> for Xx128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &&[u8; N], seed: u64) -> u128 {
        xxhash_rust::xxh3::xxh3_128_with_seed(to_bytes(x), seed)
    }
}
impl Hasher<Vec<u8>> for Xx128 {
    type H = u128;
    #[inline(always)]
    fn hash(x: &Vec<u8>, seed: u64) -> u128 {
        xxhash_rust::xxh3::xxh3_128_with_seed(to_bytes(x.as_slice()), seed)
    }
}

// Further hashes

#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for Murmur2_64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        murmur2::murmur64a(to_bytes(x), seed)
    }
}
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for FastMurmur3_128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        fastmurmur3::murmur3_x64_128(to_bytes(x), seed)
    }
}

// C. Further high quality hash functions.
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for Murmur3_128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        let mut bytes = to_bytes(x);
        murmur3::murmur3_x64_128(&mut bytes, seed as u32).unwrap()
    }
}
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for Highway64 {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        use highway::HighwayHash;
        highway::HighwayHasher::default().hash64(to_bytes(x))
    }
}
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for Highway128 {
    type H = u128;
    fn hash(x: &Key, _seed: u64) -> u128 {
        use highway::HighwayHash;
        let words = highway::HighwayHasher::default().hash128(to_bytes(x));
        unsafe { std::mem::transmute(words) }
    }
}
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for City64 {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        cityhash_102_rs::city_hash_64(to_bytes(x))
    }
}
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for City128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        cityhash_102_rs::city_hash_128_seed(to_bytes(x), seed as _)
    }
}
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for Wy64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        wyhash::wyhash(to_bytes(x), seed)
    }
}
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for Metro64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        use std::hash::Hasher;
        let mut hasher = metrohash::MetroHash64::with_seed(seed);
        hasher.write(to_bytes(x));
        hasher.finish()
    }
}
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for Metro128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        use std::hash::Hasher;
        let mut hasher = metrohash::MetroHash128::with_seed(seed);
        hasher.write(to_bytes(x));
        let (l, h) = hasher.finish128();
        (h as u128) << 64 | l as u128
    }
}
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for Spooky64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        use std::hash::Hasher;
        let mut hasher = hashers::jenkins::spooky_hash::SpookyHasher::new(seed, 0);
        hasher.write(to_bytes(x));
        hasher.finish()
    }
}
#[cfg(feature = "hashers")]
impl<Key> Hasher<Key> for Spooky128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        use std::hash::Hasher;
        let mut hasher = hashers::jenkins::spooky_hash::SpookyHasher::new(seed, 0);
        hasher.write(to_bytes(x));
        let (l, h) = hasher.finish128();
        (h as u128) << 64 | l as u128
    }
}
