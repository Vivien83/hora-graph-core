# hora-graph-wasm

Bio-inspired embedded knowledge graph engine for the browser & edge runtimes — compiled to WebAssembly via [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/).

## Install

```bash
npm install hora-graph-wasm
```

## Quick start

```js
import init, { HoraCore } from 'hora-graph-wasm';

await init();

const graph = new HoraCore();

// Add entities
const alice = graph.add_entity('person', { name: 'Alice' });
const bob = graph.add_entity('person', { name: 'Bob' });

// Add edge
graph.add_edge(alice, bob, 'knows', { since: 2024 });

// Traverse
const friends = graph.neighbors(alice, 'knows');
```

## Features

- **Knowledge graph** — entities, edges, properties, bi-temporal versioning
- **Vector search** — cosine similarity (scalar, no SIMD in WASM)
- **BM25 full-text** — Okapi BM25+ ranking
- **Hybrid search** — RRF fusion of vector + text
- **Memory model** — ACT-R activation, spreading, FSRS scheduling
- **Storage** — in-memory only (Memory backend)

## Bundle size

~166 KB gzipped (wasm + JS glue)

## License

MIT OR Apache-2.0
