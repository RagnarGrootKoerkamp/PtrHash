#![feature(type_changing_struct_update, try_blocks, slice_as_array)]

use std::{cmp::min, collections::HashMap, hint::black_box, time::Instant};

use cacheline_ef::CachelineEfVec;
use ptr_hash::{
    bucket_fn::{BucketFn, CubicEps, Linear, Optimal, Skewed, Square},
    hash::{
        FastIntHash, Gx, Gx128, GxInt, KeyHasher, NoHash, StringHash, StringHash128,
        StrongerIntHash, Xxh3, Xxh3Int, Xxh3_128,
    },
    pack::{EliasFano, MutPacked},
    stats::BucketStats,
    util::{generate_keys, generate_string_keys},
    KeyT, PtrHash, PtrHashParams, Sharding,
};
use rand::{rng, Rng, RngCore};
use rayon::iter::IntoParallelIterator;
use serde::Serialize;

/// Experiments:
/// 1. bucket sizes & evictions during construction
/// 2. construction speed, datastructure size, and query throughput for various parameters
/// 3. remap types
fn main() {
    env_logger::init();
    // bucket_fn_stats(); // <10min

    // size(); // many hours

    // remap(); // 22min

    // sharding(Sharding::Hybrid(1 << 37), "data/sharding_hybrid.json"); // 40min
    // sharding(Sharding::Memory, "data/sharding_memory.json"); // 50min

    // query_batching(); // 40min

    // query_throughput(); // 22min

    string_queries(); // 30min

    // construction_memory(); // fast
}

#[allow(unused)]
fn all() {
    bucket_fn_stats(); // <10min
    size(); // many hours
    remap(); // 12min
    sharding(Sharding::Hybrid(1 << 37), "data/sharding_hybrid.json"); // 55min
    sharding(Sharding::Memory, "data/sharding_memory.json"); // 1h
    query_batching(); // 40min
    query_throughput(); // 12min
    string_queries();
    construction_memory();
}

const SMALL_N: usize = 20_000_000;
const LARGE_N: usize = 1_000_000_000;
const NUM_QUERIES: usize = 1_000_000_000;

const PARAMS_FAST: PtrHashParams<Linear> = PtrHashParams {
    alpha: 0.99,
    lambda: 3.0,
    bucket_fn: Linear,
    // defaults...
    keys_per_shard: 1 << 31,
    sharding: Sharding::None,
    remap: true,
    single_part: false,
};

#[allow(unused)]
const PARAMS_DEFAULT: PtrHashParams<CubicEps> = PtrHashParams {
    alpha: 0.99,
    lambda: 3.5,
    bucket_fn: CubicEps,
    // defaults...
    keys_per_shard: 1 << 31,
    sharding: Sharding::None,
    remap: true,
    single_part: false,
};

const PARAMS_COMPACT: PtrHashParams<CubicEps> = PtrHashParams {
    alpha: 0.99,
    lambda: 4.0,
    bucket_fn: CubicEps,
    // defaults...
    keys_per_shard: 1 << 31,
    sharding: Sharding::None,
    remap: true,
    single_part: false,
};

#[derive(Debug, Serialize, Default)]
struct Result {
    n: usize,
    alpha: f64,
    lambda: f64,
    bucketfn: String,
    slots_per_part: usize,
    real_alpha: f64,
    construction_1: f64,
    construction_6: f64,
    pilots: f64,
    remap: f64,
    remap_type: String,
    total: f64,
    q1_phf: f64,
    q1_mphf: f64,
    q1_phf_bb: f64,
    q1_mphf_bb: f64,
    q32_phf: f64,
    q32_mphf: f64,
}

#[derive(Debug, Serialize, Default, Clone)]
struct QueryResult {
    n: usize,
    alpha: f64,
    lambda: f64,
    bucketfn: String,
    construction_6: f64,
    pilots: f64,
    remap: f64,
    remap_type: String,
    total: f64,
    batch_size: usize,
    threads: usize,
    // loop/stream/batch
    mode: String,
    q_phf: f64,
    q_mphf: f64,
    input_type: Option<String>,
    hash: Option<String>,
}

/// Collect stats on bucket sizes and number of evictions during construction.
/// Vary the bucket assignment function.
fn bucket_fn_stats() {
    type MyPtrHash<BF> = PtrHash<u64, BF, CachelineEfVec, StrongerIntHash, Vec<u8>>;
    let n = 1_000_000_000;
    let keys = &generate_keys(n);

    fn build(keys: &Vec<u64>, lambda: f64, bucket_fn: impl BucketFn) -> BucketStats {
        let params = PtrHashParams {
            lambda,
            bucket_fn,
            alpha: 0.99,
            ..PtrHashParams::default_compact()
        };
        MyPtrHash::new_with_stats(&keys, params).1
    }

    {
        let lambda = 3.5;
        let mut stats = HashMap::new();
        stats.insert("linear", build(keys, lambda, Linear));
        stats.insert("skewed", build(keys, lambda, Skewed::default()));
        stats.insert("optimal", build(keys, lambda, Optimal { eps: 1. / 256. }));
        stats.insert("square", build(keys, lambda, Square));
        stats.insert("cubic", build(keys, lambda, CubicEps));

        write(&stats, "data/bucket_fn_stats_l35.json");
    }

    {
        let lambda = 4.0;
        let mut stats = HashMap::new();
        stats.insert("cubic", build(keys, lambda, CubicEps));

        write(&stats, "data/bucket_fn_stats_l40.json");
    }
}

fn write<T: Serialize>(stats: &T, path: &str) {
    let json = serde_json::to_string(stats).unwrap();
    std::fs::write(path, json).unwrap();
}

/// Construction time&space for various lambda.
fn size() {
    fn test<R: MutPacked>(
        keys: &Vec<u64>,
        alpha: f64,
        lambda: f64,
        bucket_fn: impl BucketFn,
    ) -> Option<Result> {
        type MyPtrHash<BF, R> = PtrHash<u64, BF, R, StrongerIntHash, Vec<u8>>;

        let params = PtrHashParams {
            alpha,
            lambda,
            bucket_fn,
            ..PtrHashParams::default_compact()
        };
        eprintln!("Running {alpha} {lambda} {bucket_fn:?}");
        // Construct on 6 threads.
        let (ph, c6) = time(|| <MyPtrHash<_, R>>::try_new(&keys, params));
        let ph = ph.as_ref();

        // Space usage.
        let (pilots, remap) = ph.map(|ph| ph.bits_per_element())?;
        let total = pilots + remap;
        let r = Result {
            n: keys.len(),
            alpha,
            slots_per_part: ph.map(|ph| ph.slots_per_part()).unwrap_or_default(),
            real_alpha: ph
                .map(|ph| keys.len() as f64 / ph.max_index() as f64)
                .unwrap_or_default(),
            lambda,
            construction_6: c6,
            bucketfn: format!("{bucket_fn:?}"),
            pilots,
            remap,
            remap_type: R::name(),
            total,
            ..Result::default()
        };
        eprintln!("Result: {r:?}");
        Some(r)
    }

    let n = LARGE_N;
    let mut results = vec![];
    let keys = &generate_keys(n);
    for alpha in [0.98, 0.99, 0.995, 0.998] {
        for lambda in 27..45 {
            let lambda = lambda as f64 / 10.;
            let Some(r) = test::<Vec<u32>>(keys, alpha, lambda, Linear) else {
                break;
            };
            results.push(r);
            eprintln!();
        }
        for lambda in 27..45 {
            let lambda = lambda as f64 / 10.;
            let Some(r) = (if alpha >= 0.995 {
                test::<Vec<u32>>(keys, alpha, lambda, CubicEps)
            } else {
                test::<CachelineEfVec>(keys, alpha, lambda, CubicEps)
            }) else {
                break;
            };
            results.push(r);
            eprintln!();
        }
    }
    write(&results, &format!("data/size.json"));
}

/// Construction memory for compact params.
fn construction_memory() {
    let n = 1_000_000_000;
    let keys = &generate_keys(n);

    type MyPtrHash = PtrHash<u64, CubicEps, CachelineEfVec, StrongerIntHash, Vec<u8>>;

    let params = PARAMS_COMPACT;
    // Construct on 6 threads.
    let (ph, _c6) = time(|| <MyPtrHash>::try_new(&keys, params));
    let ph = ph.unwrap();
    black_box(ph);
}

/// Collect stats on size and query speed, varying alpha and lambda.
fn remap() {
    /// Return:
    /// Construction time (1 and 6 threads)
    /// Space (pilots, remap, total)
    /// Query throughput (32 streaming)
    fn test<R: MutPacked + Send>(
        keys: &Vec<u64>,
        alpha: f64,
        lambda: f64,
        bucket_fn: impl BucketFn + Send,
    ) -> Result {
        type MyPtrHash<BF, R> = PtrHash<u64, BF, R, StrongerIntHash, Vec<u8>>;

        let params = PtrHashParams {
            alpha,
            lambda,
            bucket_fn,
            ..PtrHashParams::default_compact()
        };

        // Construct on 6 threads.
        let (ph, c6) = time(|| <MyPtrHash<_, R>>::new(&keys, params));

        // Space usage.
        let (pilots, remap) = ph.bits_per_element();
        let total = pilots + remap;

        // Single threaded query throughput, non-minimal and minimal.
        let q1_phf = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                sum += ph.index_no_remap(key);
            }
            sum
        });
        let q1_mphf = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                sum += ph.index(key);
            }
            sum
        });
        let q1_phf_bb = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                black_box(());
                sum += ph.index_no_remap(key);
            }
            sum
        });
        let q1_mphf_bb = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                black_box(());
                sum += ph.index(key);
            }
            sum
        });
        let q32_phf = time_query(keys, || ph.index_stream::<32, false, _>(keys));
        let q32_mphf = time_query(keys, || ph.index_stream::<32, true, _>(keys));

        let r = Result {
            n: keys.len(),
            alpha,
            lambda,
            slots_per_part: ph.slots_per_part(),
            real_alpha: keys.len() as f64 / ph.max_index() as f64,
            construction_1: 0.,
            construction_6: c6,
            bucketfn: format!("{bucket_fn:?}"),
            remap_type: R::name(),
            pilots,
            remap,
            total,
            q1_phf,
            q1_mphf,
            q1_phf_bb,
            q1_mphf_bb,
            q32_phf,
            q32_mphf,
        };
        eprintln!("{r:?}");

        r
    }

    let n = 1_000_000_000;
    let mut results = vec![];
    let keys = &generate_keys(n);

    // SIMPLE
    {
        let alpha = 0.99;
        let lambda = 3.0;
        results.push(test::<Vec<u32>>(keys, alpha, lambda, Linear));
        results.push(test::<CachelineEfVec>(keys, alpha, lambda, Linear));
        results.push(test::<EliasFano>(keys, alpha, lambda, Linear));
    }
    // DEFAULT
    {
        let alpha = 0.99;
        let lambda = 3.5;
        results.push(test::<Vec<u32>>(keys, alpha, lambda, CubicEps));
        results.push(test::<CachelineEfVec>(keys, alpha, lambda, CubicEps));
        results.push(test::<EliasFano>(keys, alpha, lambda, CubicEps));
    }
    // COMPACT
    {
        let alpha = 0.99;
        let lambda = 4.0;
        results.push(test::<Vec<u32>>(keys, alpha, lambda, CubicEps));
        results.push(test::<CachelineEfVec>(keys, alpha, lambda, CubicEps));
        results.push(test::<EliasFano>(keys, alpha, lambda, CubicEps));
    }
    write(&results, &format!("data/remap.json"));
}

fn sharding(sharding: Sharding, path: &str) {
    let n = 100_000_000_000 / 2;
    let range = 0..n as u64;
    let keys = range.into_par_iter();
    let start = Instant::now();
    let bucket_fn = CubicEps;
    type MyPtrHash = PtrHash<u64, CubicEps, CachelineEfVec, StrongerIntHash, Vec<u8>>;
    let ptr_hash = MyPtrHash::new_from_par_iter(
        n,
        keys,
        PtrHashParams {
            lambda: 3.5,
            alpha: 0.99,
            // ~16GiB of keys per shard.
            keys_per_shard: 1 << 31,
            // Max 128GiB of memory at a time.
            sharding,
            bucket_fn,
            ..PtrHashParams::default_compact()
        },
    );
    let c6 = start.elapsed().as_secs_f64();
    let (pilots, remap) = ptr_hash.bits_per_element();
    let total = pilots + remap;
    let r = Result {
        n,
        lambda: 3.5,
        alpha: 0.99,
        construction_6: c6,
        bucketfn: format!("{:?}", bucket_fn),
        pilots,
        remap,
        total,
        ..Result::default()
    };

    eprintln!("Sharding {sharding:?}: {c6}s",);
    write(&r, path);
}

fn query_batching() {
    fn test(keys: &Vec<u64>, params: PtrHashParams<impl BucketFn>, rs: &mut Vec<QueryResult>) {
        type MyPtrHash<BF> = PtrHash<u64, BF, Vec<u32>, StrongerIntHash, Vec<u8>>;
        eprintln!("Building {params:?}");
        // Construct on 6 threads.
        let (ph, c6) = time(|| MyPtrHash::new(&keys, params));

        // Space usage.
        let (pilots, remap) = ph.bits_per_element();
        let total = pilots + remap;

        let r0 = QueryResult {
            n: keys.len(),
            alpha: params.alpha,
            lambda: params.lambda,
            construction_6: c6,
            bucketfn: format!("{:?}", params.bucket_fn),
            pilots,
            remap,
            total,
            remap_type: "none".to_string(),
            ..Default::default()
        };

        let q_phf = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                sum += ph.index_no_remap(key);
            }
            sum
        });

        let r = QueryResult {
            batch_size: 0,
            mode: "loop".to_string(),
            q_phf,
            ..r0.clone()
        };
        eprintln!("Result: {r:?}");
        rs.push(r.clone());

        let q_phf = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                black_box(());
                sum += ph.index_no_remap(key);
            }
            sum
        });

        let r = QueryResult {
            batch_size: 0,
            mode: "loop_bb".to_string(),
            q_phf,
            ..r0.clone()
        };
        eprintln!("Result: {r:?}");
        rs.push(r.clone());

        fn batch<const A: usize, BF: BucketFn>(
            ph: &PtrHash<u64, BF, Vec<u32>, StrongerIntHash, Vec<u8>>,
            keys: &Vec<u64>,
            r: &QueryResult,
            rs: &mut Vec<QueryResult>,
        ) {
            let stream = time_query(keys, || ph.index_stream::<A, false, _>(keys));
            // Somehow, index_batch has very weird scaling behaviour in A.
            // index_batch2 *does* improve as A increases, and so we use that one instead.
            // let batch = time_query(keys, || ph.index_batch_exact::<A, false>(keys));
            let batch2 = time_query(keys, || ph.index_batch_exact2::<A, false>(keys));
            rs.push(QueryResult {
                batch_size: A,
                mode: "stream".to_string(),
                q_phf: stream,
                ..r.clone()
            });
            eprintln!("Result: {:?}", rs.last().unwrap());
            // rs.push(QueryResult {
            //     batch_size: A,
            //     mode: "batch".to_string(),
            //     q_phf: batch,
            //     ..r.clone()
            // });
            // eprintln!("Result: {:?}", rs.last().unwrap());
            rs.push(QueryResult {
                batch_size: A,
                mode: "batch2".to_string(),
                q_phf: batch2,
                ..r.clone()
            });
            eprintln!("Result: {:?}", rs.last().unwrap());
        }
        batch::<1, _>(&ph, keys, &r, rs);
        batch::<2, _>(&ph, keys, &r, rs);
        batch::<3, _>(&ph, keys, &r, rs);
        batch::<4, _>(&ph, keys, &r, rs);
        batch::<5, _>(&ph, keys, &r, rs);
        batch::<6, _>(&ph, keys, &r, rs);
        batch::<7, _>(&ph, keys, &r, rs);
        batch::<8, _>(&ph, keys, &r, rs);
        batch::<10, _>(&ph, keys, &r, rs);
        batch::<12, _>(&ph, keys, &r, rs);
        batch::<14, _>(&ph, keys, &r, rs);
        batch::<16, _>(&ph, keys, &r, rs);
        batch::<20, _>(&ph, keys, &r, rs);
        batch::<24, _>(&ph, keys, &r, rs);
        batch::<28, _>(&ph, keys, &r, rs);
        batch::<32, _>(&ph, keys, &r, rs);
        batch::<40, _>(&ph, keys, &r, rs);
        batch::<48, _>(&ph, keys, &r, rs);
        batch::<56, _>(&ph, keys, &r, rs);
        batch::<64, _>(&ph, keys, &r, rs);
    }

    let mut results = vec![];
    for n in [SMALL_N, LARGE_N] {
        let keys = &generate_keys(n);

        test(keys, PARAMS_FAST, &mut results);
        // test(keys, PARAMS_DEFAULT, &mut results); // Identical to compact
        test(keys, PARAMS_COMPACT, &mut results);
    }
    write(&results, "data/query_batching.json");
}

fn time_query<K: KeyT, I: Iterator<Item = usize>>(keys: &[K], f: impl Fn() -> I) -> f64 {
    time_query_f(keys, || f().sum::<usize>())
}

fn time_query_f<K: KeyT>(keys: &[K], f: impl Fn() -> usize) -> f64 {
    let loops = NUM_QUERIES / keys.len();
    let t = time(|| black_box((0..loops).map(|_| f()).sum::<usize>())).1;
    // convert to ns/key
    t * 1_000_000_000. / (loops * keys.len()) as f64
}

fn time_query_parallel<'k, I: Iterator<Item = usize>>(
    threads: usize,
    keys: &'k Vec<u64>,
    f: impl Fn(&'k [u64]) -> I + Send + Sync,
) -> f64 {
    time_query_parallel_f(
        threads,
        keys,
        #[inline(always)]
        |keys| f(keys).sum::<usize>(),
    )
}

fn time_query_parallel_f<'k>(
    threads: usize,
    keys: &'k Vec<u64>,
    f: impl Fn(&'k [u64]) -> usize + Send + Sync,
) -> f64 {
    let loops = NUM_QUERIES / keys.len();
    let chunk_size = keys.len().div_ceil(threads);

    let t = time(move || {
        rayon::scope(|scope| {
            for thread_idx in 0..threads {
                let f = &f;
                scope.spawn(move |_| {
                    let mut sum = 0;

                    for l in 0..loops {
                        let idx = (thread_idx + l) % threads;
                        let start_idx = idx * chunk_size;
                        let end = min((idx + 1) * chunk_size, keys.len());
                        sum += f(&keys[start_idx..end]);
                    }
                    black_box(sum);
                });
            }
        });
    })
    .1;
    // convert to ns/key
    t * 1_000_000_000. / (loops * keys.len()) as f64
}

fn query_throughput() {
    fn test<R: MutPacked>(
        keys: &Vec<u64>,
        params: PtrHashParams<impl BucketFn>,
        rs: &mut Vec<QueryResult>,
    ) {
        type MyPtrHash<BF, R> = PtrHash<u64, BF, R, StrongerIntHash, Vec<u8>>;
        eprintln!("Building {params:?}");
        // Construct on 6 threads.
        let (ph, c6) = time(|| MyPtrHash::<_, R>::new(&keys, params));

        // Space usage.
        let (pilots, remap) = ph.bits_per_element();
        let total = pilots + remap;

        let r0 = QueryResult {
            n: keys.len(),
            alpha: params.alpha,
            lambda: params.lambda,
            construction_6: c6,
            bucketfn: format!("{:?}", params.bucket_fn),
            pilots,
            remap,
            total,
            remap_type: R::name(),
            ..Default::default()
        };

        // When n is small, queries perfectly scale to >1 threads anyway.
        let max_threads = 6;
        for threads in 1..=max_threads {
            let q_phf = time_query_parallel_f(threads, keys, |keys| {
                let mut sum = 0;
                for key in keys {
                    black_box(());
                    sum += ph.index_no_remap(key);
                }
                sum
            });
            let q_mphf = time_query_parallel_f(threads, keys, |keys| {
                let mut sum = 0;
                for key in keys {
                    black_box(());
                    sum += ph.index(key);
                }
                sum
            });

            let r = QueryResult {
                batch_size: 0,
                mode: "loop_bb".to_string(),
                q_phf,
                q_mphf,
                threads,
                ..r0.clone()
            };
            eprintln!("Result: {r:?}");
            rs.push(r.clone());

            let q_phf = time_query_parallel_f(threads, keys, |keys| {
                let mut sum = 0;
                for key in keys {
                    sum += ph.index_no_remap(key);
                }
                sum
            });
            let q_mphf = time_query_parallel_f(threads, keys, |keys| {
                let mut sum = 0;
                for key in keys {
                    sum += ph.index(key);
                }
                sum
            });

            let r = QueryResult {
                batch_size: 0,
                mode: "loop".to_string(),
                q_phf,
                q_mphf,
                threads,
                ..r0.clone()
            };
            eprintln!("Result: {r:?}");
            rs.push(r.clone());

            const A: usize = 32;
            let stream_phf =
                time_query_parallel(threads, keys, |keys| ph.index_stream::<A, false, _>(keys));
            let stream_mphf =
                time_query_parallel(threads, keys, |keys| ph.index_stream::<A, true, _>(keys));

            rs.push(QueryResult {
                batch_size: A,
                mode: "stream".to_string(),
                q_phf: stream_phf,
                q_mphf: stream_mphf,
                threads,
                ..r.clone()
            });
            eprintln!("Result: {:?}", rs.last().unwrap());
        }
    }

    let mut results = vec![];
    for n in [SMALL_N, LARGE_N] {
        let keys = &generate_keys(n);

        test::<Vec<u32>>(keys, PARAMS_FAST, &mut results);
        // test::<CachelineEfVec>(keys, PARAMS_DEFAULT, &mut results); // Identical to compact
        test::<CachelineEfVec>(keys, PARAMS_COMPACT, &mut results);
    }
    write(&results, "data/query_throughput.json");
}

fn string_queries() {
    fn test<R: MutPacked, K: KeyT, H: KeyHasher<K>>(
        keys: &Vec<K>,
        params: PtrHashParams<impl BucketFn>,
        rs: &mut Vec<QueryResult>,
    ) {
        type MyPtrHash<BF, R, K, H> = PtrHash<K, BF, R, H, Vec<u8>>;
        eprintln!("Building {params:?}");
        // Construct on 6 threads.
        let (ph, c6) = time(|| MyPtrHash::<_, R, K, H>::new(&keys, params));

        // Space usage.
        let (pilots, remap) = ph.bits_per_element();
        let total = pilots + remap;

        let r0 = QueryResult {
            n: keys.len(),
            alpha: params.alpha,
            lambda: params.lambda,
            construction_6: c6,
            bucketfn: format!("{:?}", params.bucket_fn),
            pilots,
            remap,
            total,
            remap_type: R::name(),
            input_type: Some(std::any::type_name::<K>().to_string()),
            hash: Some(std::any::type_name::<H>().to_string()),
            ..Default::default()
        };

        let q_mphf = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                black_box(());
                sum += ph.index(key);
            }
            sum
        });

        let r = QueryResult {
            batch_size: 0,
            mode: "loop_bb".to_string(),
            q_mphf,
            ..r0.clone()
        };
        eprintln!("Result: {r:?}");
        rs.push(r.clone());

        let q_mphf = time_query_f(keys, || {
            let mut sum = 0;
            for key in keys {
                sum += ph.index(key);
            }
            sum
        });

        let r = QueryResult {
            batch_size: 0,
            mode: "loop".to_string(),
            q_mphf,
            ..r0.clone()
        };
        eprintln!("Result: {r:?}");
        rs.push(r.clone());

        const A: usize = 32;
        let stream_mphf = time_query(keys, || ph.index_stream::<A, true, _>(keys));

        rs.push(QueryResult {
            batch_size: A,
            mode: "stream".to_string(),
            q_mphf: stream_mphf,
            ..r.clone()
        });
        eprintln!("Result: {:?}", rs.last().unwrap());
    }

    let mut results = vec![];
    // We can't fit 1G strings into memory, sadly.
    for n in [1000, 1000000, 100_000_000] {
        type R = Vec<u32>;

        // INT
        {
            let keys: Vec<u64> = generate_keys(n);

            test::<R, _, NoHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StrongerIntHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, FastIntHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, Xxh3Int>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StringHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StringHash128>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, GxInt>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, Gx>(&keys, PARAMS_FAST, &mut results);
        }

        // BOXED INT
        {
            let keys: Vec<Box<u64>> = generate_keys(n).into_iter().map(|k| Box::new(k)).collect();

            test::<R, _, FastIntHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StringHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StringHash128>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, Gx>(&keys, PARAMS_FAST, &mut results);
        }

        // PACKED SHORT STRING
        {
            let total_len = 10 * n + 50;
            let mut rng = rng();
            let mut string = vec![0; total_len];
            rng.fill_bytes(&mut string);
            eprintln!("String size: {total_len}");
            let mut idx = 0;
            let keys: Vec<&[u8; 10]> = (0..n)
                .map(|_| {
                    let slice = string[idx..idx + 10].as_array().unwrap();
                    idx += 10;
                    slice
                })
                .collect::<Vec<_>>();
            eprintln!("Keys size: {}", std::mem::size_of_val(keys.as_slice()));

            test::<R, _, FastIntHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StringHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StringHash128>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, Gx>(&keys, PARAMS_FAST, &mut results);
        }

        // PACKED LONG STRING
        {
            let total_len = 10 * n + 50;
            let mut rng = rng();
            let mut string = vec![0; total_len];
            rng.fill_bytes(&mut string);
            eprintln!("String size: {total_len}");
            let mut idx = 0;
            let keys: Vec<&[u8; 50]> = (0..n)
                .map(|_| {
                    let slice = string[idx..idx + 50].as_array().unwrap();
                    idx += 10;
                    slice
                })
                .collect::<Vec<_>>();
            eprintln!("Keys size: {}", std::mem::size_of_val(keys.as_slice()));

            test::<R, _, FastIntHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StringHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StringHash128>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, Gx>(&keys, PARAMS_FAST, &mut results);
        }

        // PACKED RANDOM STRING
        {
            let total_len = 10 * n + 50;
            let mut rng = rng();
            let mut string = vec![0; total_len];
            rng.fill_bytes(&mut string);
            eprintln!("String size: {total_len}");
            let mut idx = 0;
            let keys: Vec<&[u8]> = (0..n)
                .map(|_| {
                    let len = rng.random_range(10..=50);
                    let slice = &string[idx..idx + len];
                    idx += 10;
                    slice
                })
                .collect::<Vec<_>>();
            eprintln!("Keys size: {}", std::mem::size_of_val(keys.as_slice()));

            test::<R, _, FastIntHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StringHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, StringHash128>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, Gx>(&keys, PARAMS_FAST, &mut results);
        }

        // STRING
        {
            let keys: Vec<Vec<u8>> = generate_string_keys(n);

            test::<R, _, FastIntHash>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, Xxh3>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, Xxh3_128>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, Gx>(&keys, PARAMS_FAST, &mut results);
            test::<R, _, Gx128>(&keys, PARAMS_FAST, &mut results);
        }
    }
    write(&results, "data/string_queries.json");
}

fn time<T>(mut f: impl FnMut() -> T) -> (T, f64) {
    let start = Instant::now();
    let t = f();
    (t, start.elapsed().as_secs_f64())
}
