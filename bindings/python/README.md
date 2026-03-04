# hora-graph-core

Bio-inspired embedded knowledge graph engine for Python — powered by Rust via [PyO3](https://pyo3.rs) + [maturin](https://maturin.rs).

## Install

```bash
pip install hora-graph-core
```

## Quick start

```python
from hora_graph_core import HoraCore

graph = HoraCore()

# Add entities
alice = graph.add_entity("person", {"name": "Alice"})
bob = graph.add_entity("person", {"name": "Bob"})

# Add edge
graph.add_edge(alice, bob, "knows", {"since": 2024})

# Traverse
friends = graph.neighbors(alice, edge_type="knows")
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

- Linux x86_64 / aarch64
- macOS x86_64 / arm64
- Windows x86_64

Python 3.9+

## License

MIT OR Apache-2.0
