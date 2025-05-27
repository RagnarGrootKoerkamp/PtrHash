use ptr_hash::{bucket_fn::Linear, hash::Gx, pack::EliasFano, PtrHash, PtrHashParams, Sharding};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

fn main() {
    env_logger::init();

    let n = 100_000_000;
    let keys = ptr_hash::util::generate_string_keys(n);

    // String slice keys, linear bucket function, EliasFano remap, GxHash hash fucntion.
    type MyPtrHash = PtrHash<[u8], Linear, EliasFano, Gx>;
    let mut params = PtrHashParams::default();
    params.sharding = Sharding::Disk;

    let mphf =
        MyPtrHash::new_from_par_iter(keys.len(), keys.par_iter().map(|s| s.as_slice()), params);

    mphf.index(b"abc".as_slice());
}
