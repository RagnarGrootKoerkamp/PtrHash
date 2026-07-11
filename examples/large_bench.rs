//! Benchmark of constructing and querying a 10^9 table.
use ptr_hash::{
    hash::{FastIntHash, KeyHasher},
    pack::MutPacked,
    util::generate_keys,
    PtrHash, PtrHashParams,
};
use std::hint::black_box;

fn time<T>(n: usize, f: impl FnOnce() -> T) -> (f64, T) {
    let start = std::time::Instant::now();
    let r = f();
    (start.elapsed().as_secs_f64() * 1e9 / n as f64, r)
}

fn bench<
    BF: ptr_hash::bucket_fn::BucketFn,
    Hx: KeyHasher<u64>,
    F: MutPacked,
    const SINGLE_PART: bool,
    const REMAP: bool,
>(
    keys: &[u64],
    name: &str,
    params: PtrHashParams<BF>,
) {
    let n = keys.len();

    let (construction, ptr_hash) = time(n, || {
        <PtrHash<u64, BF, F, Hx, Vec<u8>, SINGLE_PART, REMAP>>::new(&keys, params)
    });
    let (pilots, remapping) = ptr_hash.bits_per_element();
    let space = pilots + remapping;
    let query = time(n, || {
        for key in keys {
            black_box(ptr_hash.index(key));
        }
    })
    .0;
    let stream = time(n, || ptr_hash.index_stream::<32, _>(keys).sum::<usize>()).0;
    let batch = time(n, || {
        black_box(
            ptr_hash
                .index_batch_exact2::<32, REMAP>(keys)
                .sum::<usize>(),
        )
    })
    .0;

    eprintln!("{name:<40}: {construction:6.1} ns/key, {pilots:4.2}+{remapping:4.2}={space:4.2} bits/key, {query:5.1} {stream:5.1} {batch:5.1}");
}

fn main() {
    env_logger::init();
    let n = 1_000_000_000;
    let keys = generate_keys(n);
    bench::<ptr_hash::bucket_fn::Linear, FastIntHash, Vec<u32>, true, false>(
        &keys,
        "FastPtrHash<FxHash>",
        PtrHashParams::default_fast(),
    );
    bench::<ptr_hash::bucket_fn::Linear, FastIntHash, Vec<u32>, true, true>(
        &keys,
        "DefaultPtrHash<FxHash>",
        PtrHashParams::default(),
    );
    bench::<ptr_hash::bucket_fn::CubicEps, FastIntHash, ptr_hash::pack::EliasFano, false, true>(
        &keys,
        "CompactPtrHash<FxHash>",
        PtrHashParams::default_compact(),
    );
}
