use epserde::prelude::*;
use ptr_hash::{PtrHash, PtrHashParams};

/// XXH3 version since the default FxHash does not implement the epserde traits.
type PtrHashXxh3 =
    PtrHash<u64, ptr_hash::bucket_fn::Linear, Vec<u32>, ptr_hash::hash::Xxh3Int, Vec<u8>>;

fn main() {
    env_logger::init();

    // Generate some random keys.
    let n = 10_000_000;
    eprintln!("Generating keys..");
    let keys = ptr_hash::util::generate_keys(n);

    // Build the datastructure.
    eprintln!("Building mphf..");
    let start = std::time::Instant::now();
    let mphf = PtrHashXxh3::new(&keys, PtrHashParams::default());
    eprintln!("construction took: {:?}", start.elapsed());

    let path = "/tmp/test.mphf";
    let mut file = std::fs::File::create(path).unwrap();
    eprintln!("\nSerializing to {path}..");
    mphf.serialize(&mut file).unwrap();
    let len = file.metadata().unwrap().len();
    eprintln!("Size: {len}, bits/key: {}\n", len as f32 * 8. / n as f32);

    // Deserialize full PtrHash
    // eprintln!("load full, without epserde..");
    // let mphf2 = <PtrHash>::deserialize_full(&mut std::fs::File::open(path).unwrap()).unwrap();

    // Ep-serde from memory.
    // eprintln!("Load into memory, with epserde..");
    // let mphf2 = <PtrHash>::load_mem(path).unwrap();

    // Ep-serde from mmap.
    eprintln!("Load into mmap, with epserde..");
    let mphf2 =
        PtrHashXxh3::load_mmap(path, Flags::RANDOM_ACCESS | Flags::TRANSPARENT_HUGE_PAGES).unwrap();

    // Ep-serde from manually read memory.
    // May fail because we require 64-byte alignment, and `fs::read` is not guaranteed to give that.
    // let b = std::fs::read(path).unwrap();
    // let mphf2 = <PtrHash>::deserialize_eps(b.as_ref()).unwrap();

    eprintln!("Testing..");
    // Get the minimal index of a key.
    let key = 0;
    let idx = mphf2.index(&key);
    assert!(idx < n);

    // Get the non-minimal index of a key. Slightly faster.
    let _idx = mphf2.index_no_remap(&key);

    // An iterator over the indices of the keys.
    // 32: number of iterations ahead to prefetch.
    // true: remap to a minimal key in [0, n).
    eprintln!("Check sum..");
    let indices = mphf2.index_stream_maybe_remap::<32, true, _>(&keys);
    assert_eq!(indices.sum::<usize>(), (n * (n - 1)) / 2);

    // Test that all items map to different indices
    eprintln!("Check taken..");
    let mut taken = vec![false; n];
    for key in keys {
        let idx = mphf2.index(&key);
        assert!(!taken[idx]);
        taken[idx] = true;
    }

    eprintln!("\n\nDO NOT FORGET TO CLEAN UP {path} !!!\n\n");
}
