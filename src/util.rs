//! Internal utilities that are only exposed for testing/benchmarking purposes.
//! Do not use externally.
use super::*;
use colored::Colorize;
use log::{trace, warn};
use rand::{rng, Rng};
use rayon::prelude::*;
use rdst::RadixSort;

/// Prefetch the given cacheline into L1 cache.
pub(crate) fn prefetch_index<T>(s: &[T], index: usize) {
    let ptr = s.as_ptr().wrapping_add(index) as *const u64;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // TODO: Put this behind a feature flag.
        // std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_LOCALITY3);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        // Do nothing.
    }
}

pub(crate) fn mul_high(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

thread_local! {
    /// TODO: Use trace! instead.
    static LOG: std::cell::Cell<bool> = std::cell::Cell::new(true);
}

pub(crate) fn log_duration(name: &str, start: Instant) -> Instant {
    if !LOG.with(|log| log.get()) {
        return start;
    }
    trace!(
        "{}",
        format!("{name:>12}: {:>13.2?}s", start.elapsed().as_secs_f32()).bold()
    );
    Instant::now()
}

pub fn generate_keys(n: usize) -> Vec<u64> {
    // TODO: Deterministic key generation.
    let start = Instant::now();
    let keys = loop {
        let start = Instant::now();
        let keys: Vec<_> = (0..n)
            .into_par_iter()
            .map_init(rng, |rng, _| rng.random())
            .collect();
        let start = log_duration("┌   gen keys", start);
        let mut keys2: Vec<_> = keys.par_iter().copied().collect();
        let start = log_duration("├      clone", start);
        keys2.radix_sort_unstable();
        let start = log_duration("├       sort", start);
        let distinct = keys2.par_windows(2).all(|w| w[0] < w[1]);
        log_duration("├ duplicates", start);
        if distinct {
            break keys;
        }
        warn!("DUPLICATE KEYS GENERATED");
    };
    log_duration("generatekeys", start);
    keys
}

pub fn generate_string_keys(n: usize) -> Vec<Vec<u8>> {
    let start = Instant::now();
    // let start = Instant::now();
    let keys: Vec<_> = (0..n)
        .into_par_iter()
        .map_init(rng, |rng, _| {
            let len = rng.random_range(10..=50);
            (0..len).map(|_| rng.random_range(1..=255)).collect()
        })
        .collect();
    log_duration("generatekeys", start);
    keys
}
