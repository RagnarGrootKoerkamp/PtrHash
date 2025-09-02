use mem_dbg::MemSize;

use crate::reduce::Reduce;

// Multiply a u128 by u64 and return the upper 64 bits of the result.
// ((lowbits * d as u128) >> 128) as u64
#[allow(unused)]
fn mul128_u64(lowbits: u128, d: u64) -> u64 {
    let bot_half = ((lowbits & u64::MAX as u128) * d as u128) >> 64; // Won't overflow
    let top_half = (lowbits >> 64) * d as u128;
    let both_halves = bot_half + top_half; // Both halves are already shifted down by 64
    (both_halves >> 64) as u64
}

/// FastMod64
/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone, Debug, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[allow(unused)]
pub struct FM64 {
    d: u64,
    m: u128,
}
impl Reduce for FM64 {
    fn new(d: usize) -> Self {
        Self {
            d: d as u64,
            m: u128::MAX / d as u128 + 1,
        }
    }
    fn reduce(self, h: u64) -> usize {
        let lowbits = self.m.wrapping_mul(h as u128);
        mul128_u64(lowbits, self.d) as usize
    }
}

/// FastMod32, using the low 32 bits of the hash.
/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone, Debug, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct FM32 {
    d: u64,
    m: u64,
}
impl Reduce for FM32 {
    fn new(d: usize) -> Self {
        assert!(d <= u32::MAX as usize);
        Self {
            d: d as u64,
            m: (u64::MAX / d as u64).wrapping_add(1),
        }
    }
    fn reduce(self, h: u64) -> usize {
        let lowbits = self.m.wrapping_mul(h as u64);
        ((lowbits as u128 * self.d as u128) >> 64) as usize
    }
}
