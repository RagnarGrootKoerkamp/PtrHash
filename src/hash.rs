use std::fmt::Debug;

use crate::KeyT;

/// A wrapper trait that supports both 64 and 128bit hashes.
pub trait Hash: Copy + Debug + Default + Send + Sync + Eq + rdst::RadixKey {
    /// Returns the low 64bits.
    fn low(&self) -> u64;
    /// Returns the high 64bits.
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

pub trait Hasher<Key>: Sync {
    type H: Hash;
    fn hash(x: &Key, seed: u64) -> Self::H;
}

pub struct Murmur;

impl<Key> Hasher<Key> for Murmur {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        murmur2::murmur64a(
            // Pass the key as a byte slice.
            unsafe {
                std::slice::from_raw_parts(x as *const Key as *const u8, std::mem::size_of::<Key>())
            },
            seed,
        )
    }
}

/// Multiply the key by a mixing constant.
pub struct MulHash;

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

/// Pass the key through unchanged.
/// Used for benchmarking.
pub struct NoHash;

impl Hasher<u64> for NoHash {
    type H = u64;
    fn hash(x: &u64, _seed: u64) -> u64 {
        *x
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct FxHash;

impl<'k, Key: KeyT<'k>> Hasher<Key> for FxHash {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        fxhash::hash64(x)
    }
}
