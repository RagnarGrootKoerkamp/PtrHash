use crate::util::mul_high;
use std::fmt::Debug;

pub trait BucketFn: Clone + Copy + Sync + Debug {
    const LINEAR: bool = false;
    const B_OUTPUT: bool = false;
    fn set_buckets_per_part(&mut self, _b: u64) {}
    fn call(&self, x: u64) -> u64;
}

#[derive(Clone, Copy, Debug)]
pub struct Linear;

impl BucketFn for Linear {
    const LINEAR: bool = true;
    fn call(&self, x: u64) -> u64 {
        x
    }
}

/// A 2-piece-wise linear function
///
/// |              .
/// |             .
/// |         ....---< gamma
/// |    .....   |
/// |....        |
/// +------------^--
///              beta
///
/// line1: y = x * (gamma / beta)
///                ~~~ slope1 ~~~
/// line2: y = x * ((1 - gamma) / (1 - beta)) + (gamma - beta) / (1 - beta)
///                ~~~~~~~~~ slope2 ~~~~~~~~~   ~~~~~~~~~~ offset ~~~~~~~~~
#[derive(Clone, Copy, Debug)]
pub struct Skewed {
    beta_f: f64,
    gamma_f: f64,
    /// buckets per part
    b: u64,
    beta: u64,
    slope1: u64,
    slope2: u64,
    neg_offset: u64,
}

impl Default for Skewed {
    fn default() -> Self {
        Skewed::new(0.6, 0.3)
    }
}

impl Skewed {
    // Map the first beta% of hashes to the first gamma% of buckets.
    pub fn new(beta: f64, gamma: f64) -> Self {
        Self {
            beta_f: beta,
            gamma_f: gamma,
            b: 0,
            beta: 0,
            slope1: 0,
            slope2: 0,
            neg_offset: 0,
        }
    }
}

impl BucketFn for Skewed {
    const B_OUTPUT: bool = true;
    fn set_buckets_per_part(&mut self, b: u64) {
        let beta = self.beta_f;
        let gamma = self.gamma_f;
        self.b = b;
        let as_u64 = |x: f64| (x * u64::MAX as f64) as u64;
        self.slope1 = mul_high(as_u64(gamma / beta), self.b);
        self.slope2 = mul_high(as_u64((1. - gamma) / (1. - beta) / 8.), self.b << 3);
        self.neg_offset = mul_high(as_u64((beta - gamma) / (1. - beta)) >> 3, self.b << 3);
        self.beta = as_u64(beta);
        eprintln!("{self:?}");
    }
    fn call(&self, x: u64) -> u64 {
        // NOTE: There is a lot of MOV/CMOV going on here.
        let is_large = x >= self.beta;
        let slope = if is_large { self.slope2 } else { self.slope1 };
        mul_high(x, slope) - is_large as u64 * self.neg_offset
        // debug_assert!(!is_large || self.p2 <= b, "p2 {} <= b {}", self.p2, b);
        // debug_assert!(!is_large || b < self.b, "b {} < p2 {}", b, self.b);
        // debug_assert!(is_large || b < self.p2, "b {} < p2 {}", b, self.p2);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Perfect {
    pub eps: f64,
}

impl BucketFn for Perfect {
    fn call(&self, x: u64) -> u64 {
        let p32 = (1u64 << 32) as f64;
        let p64 = p32 * p32;
        let p64inv = 1. / p64;
        let x = (x as f64) * p64inv;
        let y = x + (1. - self.eps) * (1. - x) * (1. - x).ln();

        (y * p64) as u64
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Square;

impl BucketFn for Square {
    fn call(&self, x: u64) -> u64 {
        mul_high(x, x)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SquareEps;

impl BucketFn for SquareEps {
    fn call(&self, x: u64) -> u64 {
        mul_high(x, x) / 128 * 127 + x / 128
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cubic;

impl BucketFn for Cubic {
    fn call(&self, x: u64) -> u64 {
        // x * x * (1 + x)/2
        mul_high(mul_high(x, x), (x >> 1) | (1 << 63))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CubicEps;

impl BucketFn for CubicEps {
    fn call(&self, x: u64) -> u64 {
        // x * x * (1 + x)/2
        mul_high(mul_high(x, x), (x >> 1) | (1 << 63)) / 128 * 127 + x / 128
    }
}

#[cfg(test)]
mod test {
    use crate::bucket_fn::BucketFn;

    #[test]
    fn test_skewed() {
        use super::Skewed;
        let mut skewed = Skewed::new(0.6, 0.3);
        skewed.set_buckets_per_part(1000000000);

        let n = 100;
        for i in 0..100 {
            let x = u64::MAX / n * i;
            let y = skewed.call(x);
            println!("{x:>20} => {y:>20}");
        }
        let beta = 11068046444225730960;
        for x in 0..50 {
            let y = skewed.call(x);
            println!("{x:>20} => {y:>20}");
        }
        for x in beta - 50..beta + 50 {
            let y = skewed.call(x);
            println!("{x:>20} => {y:>20}");
        }
        for x in u64::MAX - 50..=u64::MAX {
            let y = skewed.call(x);
            println!("{x:>20} => {y:>20}");
        }
    }
}