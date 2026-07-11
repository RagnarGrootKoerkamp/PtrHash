# Changelog

<!-- next-header -->

## git
- Support 32-bit platforms and add CI ([#26](https://github.com/RagnarGrootKoerkamp/ptrhash/pull/26))
- Move a number of dependencies behind (default-on) features and remove some
  unused deps. ([#26](https://github.com/RagnarGrootKoerkamp/ptrhash/pull/26),
  [#29](https://github.com/RagnarGrootKoerkamp/ptrhash/pull/29),  [#30](https://github.com/RagnarGrootKoerkamp/ptrhash/pull/30))
- Make things work with stable rust.
- Add new simpler `FastPtrHash`, `DefaultPtrHash`, and `CompactPtrHash` type aliases.
  - Use `FastPtrHash` if a non-minimal PHF with 1% overhead is enough.
  - Use `CompactPtrHash` if you need parallel construction and/or a minimal space usage.
- `index_no_remap` and `index_single_part` are now replaced by generics. ([#28](https://github.com/RagnarGrootKoerkamp/ptrhash/issues/28))
- Add more benchmarks to readme.

## v2.0.0-alpha.1
- Breaking: use `fastmod` for `slot_in_part` instead.
- Use Sebastiano Vigna's formula of the [epsilon-cost-sharding
  paper](https://arxiv.org/abs/2503.18397) to determine the number of slots per part.
- Clean up the default provided hashes; add GxHash for strings.
- Make `default_fast` the actual `PtrHashParams::default()`, and rename the
  previous 'default' from the paper to `default_balanced`.
- Make `RandomIntHash` the default hash function.
- Rename `IntHash` => `StrongerIntHash` and `RandomIntHash` => `FastIntHash`
  (because the latter should be the default).

**Minor.**
- Added tests
- The `epserde` feature is now not enabled by default


## v1.1
- Handle failures in CacheLineEF construction (by retrying with a new seed).
- Add construction parameter to force building only a single part.
- Breaking: some changes to the `slot_in_part` function to support
  small inputs with a single part and non-power-of-2 slots in a part.
  (Changed again for v2 -- sorry.)

**Minor.**
- Improved docs
- Mix the seed into FxHash.

## evals branch
- Added GxHash for hashing strings.

## v1.0
- Initial version.
