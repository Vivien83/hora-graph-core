# Performance Report — hora-graph-core v1.0

Environment: Apple M4, macOS, Rust 1.78+ (release profile).
Measured with Criterion 0.5 (20-sample groups) + ad-hoc 1M test.

## Summary

| Benchmark | Target | Measured | Status |
|-----------|--------|----------|--------|
| Insert 1M entities | < 5s | **1.13s** | PASS |
| Get entity by ID | < 50ns | **~240ns** | WARN |
| Add fact (edge) | — | **~219ns** | — |
| Batch 100K+500K | < 500ms | **~396ms** | PASS |
| BFS 1-hop (100K) | — | **~964ns** | — |
| BFS 2-hop (100K) | — | **~8.3µs** | — |
| BFS 3-hop (100K) | < 5ms | **~97µs** | PASS |
| Timeline (100 facts) | — | **~3.7µs** | — |
| facts_at (100K edges) | — | **~2.7ms** | — |
| Cosine 384d (NEON) | — | **~31ns** | — |
| Cosine 384d (scalar) | — | **~241ns** | — |
| Vector top-100 / 1K | — | **~155µs** | — |
| Vector top-100 / 10K | — | **~4.4ms** | — |
| Vector top-100 / 100K | < 5ms | **~50ms** | WARN |
| BM25 top-10 / 1K | — | **~34µs** | — |
| BM25 top-10 / 10K | — | **~317µs** | — |
| BM25 top-10 / 100K | < 2ms | **~3.7ms** | WARN |
| Hybrid both / 1K | — | **~204µs** | — |
| Hybrid both / 10K | < 10ms | **~3.8ms** | PASS |
| Flush 1M | — | **~128ms** | — |
| Open 1M (simple format) | < 50ms | **~559ms** | WARN |
| Dream cycle (10K) | < 1s (100K) | **~110s** (10K) | FAIL |
| File size (1M entities) | < 2 GB | **38 MB** | PASS |
| npm package (source) | < 5 MB | **9.2 KB** | PASS |
| WASM bundle (gzip) | < 500 KB | **69 KB** | PASS |

**7 PASS / 4 WARN / 1 FAIL**

## Notes on WARN / FAIL items

### Get entity ~240ns (target < 50ns)

HashMap lookup with u64 keys. The 50ns target was based on direct-index
access. Current ~240ns is still sub-microsecond and acceptable for all
use cases. Could be improved with a `Vec<Option<Entity>>` indexed by ID
if needed.

### Vector search 100K ~50ms (target < 5ms)

Brute-force linear scan. Expected for a zero-dependency engine without an
ANN index (HNSW, IVF). The 384-dimension SIMD cosine itself runs at ~31ns
per pair; the bottleneck is scanning 100K candidates. A future HNSW index
would bring this under 5ms.

### BM25 100K ~3.7ms (target < 2ms)

Close to target. The in-memory inverted index is functional but
unoptimized (no skip lists, no SIMD). A posting-list compression pass
could halve latency.

### Open 1M ~559ms (target < 50ms)

The simple `.hora` format deserializes everything into memory on open.
The 50ms target assumes mmap-based access (available in the embedded
engine via `src/storage/embedded/mmap.rs`). For the simple format,
559ms for 1M entities is acceptable.

### Dream cycle: O(n²) memory linking

`dream_cycle()` with default config creates temporal co-creation links
between all entity pairs within a session. For 10K entities with a
single session, this produces ~100M links. This is an algorithmic issue
in the linking step, not a performance regression.

**Mitigation options (future):**
- Window-based linking (only link entities within a time window)
- k-nearest linking (only link top-k most related entities)
- Batch/async dream cycle for large graphs
- Disable linking by default in `DreamCycleConfig`

## Comparison context

| Engine | Insert 1M | BFS 3-hop | Vector 100K |
|--------|-----------|-----------|-------------|
| **hora-graph-core** | **1.13s** | **97µs** | **50ms** |
| Neo4j (JVM) | ~10-30s | ~1-10ms | N/A (plugin) |
| CozoDB (Rust) | ~2-5s | ~200µs | ~10ms (HNSW) |
| Kuzu (C++) | ~1-3s | ~50µs | N/A |

hora-graph-core is competitive on insert and traversal. Vector search
requires an ANN index to match engines with HNSW.
