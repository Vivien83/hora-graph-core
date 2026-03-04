<div align="center">

# hora-graph-core

**Bio-inspired embedded knowledge graph engine in pure Rust.**

*Your memory never sleeps.*

[![CI](https://github.com/Vivien83/hora-graph-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Vivien83/hora-graph-core/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![v1.0.0](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/Vivien83/hora-graph-core/releases/tag/v1.0.0)
[![Tests](https://img.shields.io/badge/tests-310%20passing-brightgreen.svg)](#)

---

A knowledge graph that **remembers like a brain**. Built from neuroscience research on memory formation, activation decay, and sleep consolidation. Zero runtime dependencies. Ships as a single Rust crate with bindings for 5 languages.

[Developer Guide](docs/GUIDE.md) &#183; [Guide FR](docs/GUIDE-FR.md) &#183; [Performance](docs/PERFORMANCE.md) &#183; [Releases](https://github.com/Vivien83/hora-graph-core/releases)

</div>

---

## Why hora-graph-core?

Most graph databases store and retrieve. **hora-graph-core thinks.**

It models how human memory actually works — facts decay over time, frequently accessed memories strengthen, sleep consolidates what matters and forgets what doesn't. This isn't a metaphor: the engine implements peer-reviewed models from cognitive science.

| What you get | How it works |
|:---|:---|
| **Facts that know when they're true** | Bi-temporal edges with world-time validity + system-time lineage |
| **Search that combines meaning and text** | SIMD vector cosine + BM25+ full-text, fused with Reciprocal Rank Fusion |
| **Memory that strengthens with use** | ACT-R base-level learning with Petrov decay approximation |
| **Automatic forgetting of noise** | Rac1-inspired dark nodes prune low-activation entities |
| **Sleep-like consolidation** | 6-step dream cycle: SHY downscaling, replay, CLS transfer, linking |
| **Crash-safe persistence** | Page allocator + B+ tree + WAL + mmap with full recovery |
| **One crate, five languages** | Rust, Node.js, Python, WebAssembly, C |

---

## Performance

Benchmarked on Apple M3 with Criterion. All numbers are single-threaded.

```
add_entity .............. 132 ns    Insert 1M entities ...... 1.3 s
get_entity ..............  42 ns    Open 1M from file ....... 94 ms
cosine 384-dim (NEON) ...  32 ns    BFS 3-hop over 100K .... 143 us
BM25 search over 100K .. 668 us    Dream cycle 10K ........  26 ms
```

<details>
<summary><strong>What do these numbers mean?</strong></summary>

- **42 ns entity lookup** — faster than a HashMap miss. The B+ tree stays hot in L1 cache.
- **32 ns cosine similarity** — NEON SIMD on ARM, AVX2 on x86. No BLAS dependency.
- **668 us full-text search over 100K docs** — inverted index with IDF caching and stop-word elimination.
- **26 ms dream cycle** — a full 6-step memory consolidation pass over 10,000 entities, including SHY downscaling, interleaved replay, CLS transfer, and semantic linking.

</details>

---

## Quick Start

### Rust

```toml
# Cargo.toml
[dependencies]
hora-graph-core = { git = "https://github.com/Vivien83/hora-graph-core.git", tag = "v1.0.0" }
```

```rust
use hora_graph_core::{HoraCore, HoraConfig, TraverseOpts};

fn main() -> hora_graph_core::Result<()> {
    let mut hora = HoraCore::new(HoraConfig::default())?;

    // Create entities
    let alice = hora.add_entity("person", "Alice", None, None)?;
    let bob   = hora.add_entity("person", "Bob",   None, None)?;

    // Add a fact (bi-temporal directed edge)
    hora.add_fact(alice, bob, "knows", "Met at RustConf", Some(0.9))?;

    // Traverse the graph
    let result = hora.traverse(alice, TraverseOpts { depth: 3 })?;
    assert!(result.entity_ids.contains(&bob));

    // BM25 full-text search
    let hits = hora.text_search("Alice", 5)?;
    assert!(!hits.is_empty());

    // Persist to disk
    let mut hora = HoraCore::open("graph.hora", HoraConfig::default())?;
    let _id = hora.add_entity("demo", "test", None, None)?;
    hora.flush()?;

    Ok(())
}
```

### Node.js

```js
const { HoraCore } = require('@hora-engine/graph-core');

const g = HoraCore.newMemory();
const alice = g.addEntity('person', 'Alice');
const bob   = g.addEntity('person', 'Bob');
g.addFact(alice, bob, 'knows', 'Met at RustConf', 0.9);

const result = g.traverse(alice, { depth: 3 });
console.log(`Found ${result.entityIds.length} entities`);
```

### Python

```python
from hora_graph_core import HoraGraph

g = HoraGraph()
alice = g.add_entity("person", "Alice")
bob   = g.add_entity("person", "Bob")
g.add_fact(alice, bob, "knows", "Met at RustConf", 0.9)

result = g.traverse(alice, depth=3)
print(f"Found {len(result['entity_ids'])} entities")
```

<details>
<summary><strong>WebAssembly</strong></summary>

```js
import init, { HoraWasm } from 'hora-graph-wasm';

await init();
const g = new HoraWasm();
const id = g.addEntity('person', 'Alice');
const result = g.traverse(id, 3);
```

166 KB gzipped. Memory backend only. Runs in browsers and edge runtimes.

</details>

<details>
<summary><strong>C</strong></summary>

```c
#include "hora_graph_core.h"

HoraCore *g = hora_new_memory();
uint64_t alice = hora_add_entity(g, "person", "Alice");
uint64_t bob   = hora_add_entity(g, "person", "Bob");
hora_add_fact(g, alice, bob, "knows", "Met at RustConf", 0.9);
hora_free(g);
```

Auto-generated header via `cbindgen`. Static and dynamic linking supported.

</details>

---

## Features

### Graph Engine
- **Bi-temporal facts** — every edge carries `valid_at`/`invalid_at` (world-time) and `created_at` (system-time)
- **Graph traversal** — BFS with configurable depth, edge filtering, and temporal windowing
- **Triple deduplication** — name-exact + Jaccard token + cosine embedding similarity
- **Transactions** — multi-statement `begin`/`commit`/`rollback` with full isolation

### Search
- **SIMD vector search** — NEON (ARM) and AVX2 (x86) cosine similarity, brute-force top-k
- **BM25+ full-text** — inverted index with stop words, IDF caching, configurable k1/b
- **Hybrid search** — Reciprocal Rank Fusion combining vector + text results

### Bio-Inspired Memory
- **ACT-R activation** — base-level learning with Petrov decay approximation
- **Spreading activation** — fan-effect weighted propagation through edges
- **Reconsolidation** — memory destabilization window (Nader 2000 model)
- **Dark nodes** — Rac1-inspired active forgetting of low-activation entities
- **FSRS scheduling** — spaced repetition with retrievability + stability tracking

### Dream Cycle (6-step consolidation)
1. **SHY downscaling** — synaptic homeostasis: scale down all activations
2. **Interleaved replay** — stochastic replay of recent + older memories
3. **CLS transfer** — complementary learning systems: consolidate episodic to semantic
4. **Memory linking** — create new edges between co-activated entities
5. **Dark node pruning** — remove entities below activation threshold
6. **Stats collection** — report what was replayed, linked, and pruned

### Storage
- **Memory** — in-process, zero allocation overhead
- **Embedded file** — page allocator + B+ tree + WAL + mmap, crash recovery, compaction
- **SQLite** — FTS5 full-text, single-file portability
- **PostgreSQL** — tsvector search, production-grade durability

---

## Architecture

```
hora-graph-core
├── src/
│   ├── lib.rs              HoraCore — unified API (~40 public methods)
│   ├── error.rs            HoraError enum
│   ├── core/
│   │   ├── types.rs        Entity, Edge, Episode, SearchHit
│   │   ├── entity.rs       Entity CRUD
│   │   ├── edge.rs         Bi-temporal edge management
│   │   ├── episode.rs      Episode lifecycle
│   │   └── dedup.rs        Triple deduplication (3 strategies)
│   ├── memory/
│   │   ├── activation.rs   ACT-R base-level learning
│   │   ├── spreading.rs    Fan-effect spreading activation
│   │   ├── reconsolidation.rs  Nader 2000 reconsolidation
│   │   ├── dark_nodes.rs   Rac1 active forgetting
│   │   ├── fsrs.rs         Spaced repetition scheduling
│   │   └── consolidation.rs   6-step dream cycle
│   ├── search/
│   │   ├── vector.rs       SIMD cosine (NEON + AVX2)
│   │   ├── bm25.rs         BM25+ inverted index
│   │   └── hybrid.rs       Reciprocal Rank Fusion
│   └── storage/
│       ├── traits.rs       StorageOps trait (Send)
│       ├── memory.rs       In-memory backend
│       ├── sqlite.rs       SQLite + FTS5
│       ├── pg.rs           PostgreSQL + tsvector
│       └── embedded/
│           ├── page.rs     Page allocator (CRC32, freelist)
│           ├── btree.rs    B+ tree index
│           ├── wal.rs      Write-ahead log
│           ├── mmap.rs     Memory-mapped reader
│           ├── recovery.rs Crash recovery
│           ├── compaction.rs  Log compaction
│           └── tx.rs       Transaction manager
├── bindings/
│   ├── node/       napi-rs v2
│   ├── python/     PyO3 0.22 + maturin
│   ├── wasm/       wasm-bindgen
│   └── c/          cbindgen 0.27
├── benches/        Criterion 0.5 benchmarks
└── tests/          310 tests + conformance suite
```

---

## Documentation

| Resource | Description |
|:---|:---|
| **[Developer Guide (EN)](docs/GUIDE.md)** | Complete API reference with runnable examples — 784 lines, 12 sections |
| **[Guide Developpeur (FR)](docs/GUIDE-FR.md)** | Reference API complete avec exemples — 765 lignes, 12 sections |
| **[Performance Report](docs/PERFORMANCE.md)** | Criterion benchmarks, methodology, and analysis |

---

## Building from source

```bash
git clone https://github.com/Vivien83/hora-graph-core.git
cd hora-graph-core

# Run all tests
cargo test

# Run benchmarks
cargo bench

# Build with SQLite backend
cargo build --features sqlite

# Build with PostgreSQL backend
cargo build --features postgres
```

**Minimum Rust version:** 1.70

---

## Neuroscience References

The memory subsystem implements models from published research:

| Model | Paper | Module |
|:---|:---|:---|
| ACT-R base-level learning | Anderson & Lebiere (1998) | `memory/activation.rs` |
| Petrov decay approximation | Petrov (2006) | `memory/activation.rs` |
| Spreading activation | Anderson (1983) | `memory/spreading.rs` |
| Memory reconsolidation | Nader, Schafe & Le Doux (2000) | `memory/reconsolidation.rs` |
| Rac1 active forgetting | Shuai et al. (2010) | `memory/dark_nodes.rs` |
| FSRS spaced repetition | Ye (2023) | `memory/fsrs.rs` |
| Synaptic homeostasis (SHY) | Tononi & Cirelli (2003) | `memory/consolidation.rs` |
| Complementary learning systems | McClelland et al. (1995) | `memory/consolidation.rs` |

---

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

---

<div align="center">

**[hora-graph-core](https://github.com/Vivien83/hora-graph-core)** is built with care in pure Rust.

*17,500+ lines | 310 tests | zero dependencies | zero unsafe*

</div>
