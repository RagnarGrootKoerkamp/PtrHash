//! Benchmark query throughput of the default configurations.
use ptr_hash::{
    hash::{FastIntHash, NoHash},
    util::generate_keys,
    CompactPtrHash, DefaultPtrHash, FastPtrHash, PtrHashParams,
};
use std::hint::black_box;

fn time(n: usize, name: &str, f: impl FnOnce()) {
    let start = std::time::Instant::now();
    f();
    let time = start.elapsed().as_secs_f64() / n as f64 * 1e9;
    eprintln!("{name:<40}: {time:.1} ns/key");
}

fn main() {
    env_logger::init();
    for n in [100_000, 1_000_000, 10_000_000, 100_000_000] {
        eprintln!("Testing n = {}", n);
        let keys = generate_keys(n);

        let ptr_hash = FastPtrHash::<NoHash, _>::new(&keys, PtrHashParams::default_fast());
        time(n, "FastPtrHash<NoHash>", || {
            for key in &keys {
                black_box(ptr_hash.index(key));
            }
        });

        let ptr_hash = DefaultPtrHash::<NoHash, _>::new(&keys, PtrHashParams::default());
        time(n, "DefaultPtrHash<NoHash>", || {
            for key in &keys {
                black_box(ptr_hash.index(key));
            }
        });

        let ptr_hash = CompactPtrHash::<NoHash, _>::new(&keys, PtrHashParams::default_compact());
        time(n, "CompactPtrHash<NoHash>", || {
            for key in &keys {
                black_box(ptr_hash.index(key));
            }
        });

        let ptr_hash = FastPtrHash::<FastIntHash, _>::new(&keys, PtrHashParams::default_fast());
        time(n, "FastPtrHash<FxHash>", || {
            for key in &keys {
                black_box(ptr_hash.index(key));
            }
        });

        let ptr_hash = DefaultPtrHash::<FastIntHash, _>::new(&keys, PtrHashParams::default());
        time(n, "DefaultPtrHash<FxHash>", || {
            for key in &keys {
                black_box(ptr_hash.index(key));
            }
        });

        let ptr_hash =
            CompactPtrHash::<FastIntHash, _>::new(&keys, PtrHashParams::default_compact());
        time(n, "CompactPtrHash<FxHash>", || {
            for key in &keys {
                black_box(ptr_hash.index(key));
            }
        });
        eprintln!();
    }
}
