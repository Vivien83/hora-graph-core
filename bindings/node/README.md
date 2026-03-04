# hora-graph-core

Bio-inspired embedded knowledge graph engine for Node.js — powered by Rust via [napi-rs](https://napi.rs).

## Install

```bash
npm install hora-graph-core
```

## Quick start

```js
const { HoraCore } = require('hora-graph-core');

const graph = new HoraCore();

// Add entities
const alice = graph.addEntity('person', { name: 'Alice' });
const bob = graph.addEntity('person', { name: 'Bob' });

// Add edge
graph.addEdge(alice, bob, 'knows', { since: 2024 });

// Traverse
const friends = graph.neighbors(alice, { edgeType: 'knows' });
```

## Features

- **Knowledge graph** — entities, edges, properties, bi-temporal versioning
- **Vector search** — SIMD-accelerated cosine similarity
- **BM25 full-text** — Okapi BM25+ ranking
- **Hybrid search** — RRF fusion of vector + text
- **Memory model** — ACT-R activation, spreading, FSRS scheduling
- **Consolidation** — SHY downscaling, interleaved replay, dream cycle
- **Storage** — embedded pages, SQLite, PostgreSQL

## Supported platforms

| OS      | Arch    | Libc  |
|---------|---------|-------|
| macOS   | x64     | —     |
| macOS   | arm64   | —     |
| Linux   | x64     | glibc |
| Linux   | x64     | musl  |
| Linux   | arm64   | glibc |
| Linux   | arm64   | musl  |
| Windows | x64     | MSVC  |
| Windows | arm64   | MSVC  |

## License

MIT OR Apache-2.0
