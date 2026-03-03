# hora-graph-core

Bio-inspired embedded knowledge graph engine in pure Rust.

Zero runtime dependencies. Bi-temporal facts. SIMD vector search. BM25+ full-text.
ACT-R memory activation. Dream cycle consolidation. Crash-safe embedded storage.

## Quick Start (Rust)

```rust
use hora_graph_core::{HoraCore, HoraConfig};

let mut hora = HoraCore::new(HoraConfig::default()).unwrap();

// Create entities
let alice = hora.add_entity("person", "Alice", None, None).unwrap();
let bob   = hora.add_entity("person", "Bob",   None, None).unwrap();

// Add a fact (bi-temporal edge)
hora.add_fact(alice, "knows", bob, None, None, None).unwrap();

// Traverse the graph
let result = hora.traverse(alice, &Default::default()).unwrap();
assert!(result.entities.contains_key(&bob));

// Persist to disk
hora.flush_to("graph.hora").unwrap();
```

## Quick Start (Node.js)

```js
const { HoraGraph } = require('@hora-engine/graph-core');

const g = new HoraGraph();
const alice = g.addEntity('person', 'Alice');
const bob   = g.addEntity('person', 'Bob');
g.addFact(alice, 'knows', bob);
```

## Quick Start (Python)

```python
from hora_graph_core import HoraGraph

g = HoraGraph()
alice = g.add_entity("person", "Alice")
bob   = g.add_entity("person", "Bob")
g.add_fact(alice, "knows", bob)
```

## Features

| Feature | Description |
|---------|-------------|
| Bi-temporal facts | `valid_at` / `invalid_at` world-time + `created_at` system-time |
| SIMD vector search | NEON (ARM) and AVX2 (x86) cosine similarity, brute-force top-k |
| BM25+ full-text | Inverted index with stop words, IDF caching, top-k extraction |
| Hybrid search | Reciprocal Rank Fusion combining vector + BM25 results |
| Triple dedup | Name-exact, Jaccard token, cosine embedding similarity |
| ACT-R activation | Base-level learning with Petrov decay approximation |
| Spreading activation | Fan-effect weighted propagation through edges |
| Reconsolidation | Memory destabilization window (Nader 2000 model) |
| Dark nodes | Rac1-inspired active forgetting of low-activation entities |
| FSRS scheduling | Spaced repetition with retrievability + stability tracking |
| Dream cycle | 6-step consolidation: SHY downscaling, replay, CLS, linking |
| Embedded storage | Page allocator, B+ tree, WAL, mmap, crash recovery, compaction |
| Transactions | Multi-statement begin/commit/rollback |
| Multi-backend | Memory, embedded file, SQLite (FTS5), PostgreSQL (tsvector) |

## Performance (Apple M3)

| Operation | Result |
|-----------|--------|
| `add_entity` | ~132 ns |
| `get_entity` | ~42 ns |
| Cosine 384d (NEON) | ~32 ns |
| BM25 search 100K | ~668 us |
| BFS 3-hop 100K | ~143 us |
| Insert 1M entities | ~1.3 s |
| Open 1M (file) | ~94 ms |
| Dream cycle 10K | ~26 ms |

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for full report.

## Bindings

| Language | Package | Path |
|----------|---------|------|
| Rust | `hora-graph-core` | `./` |
| Node.js | `@hora-engine/graph-core` | `bindings/node/` |
| Python | `hora-graph-core` | `bindings/python/` |
| WASM | `hora-graph-wasm` | `bindings/wasm/` |
| C | `hora-graph-ffi` | `bindings/c/` |

## Architecture

```
src/
  core/       types, entity, edge, episode, dedup
  memory/     activation, spreading, reconsolidation, dark_nodes, fsrs, consolidation
  search/     vector (SIMD), bm25, hybrid (RRF)
  storage/    traits, memory, format, embedded/ (page, btree, wal, mmap, recovery, compaction, tx)
  lib.rs      HoraCore API
  error.rs    HoraError
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
