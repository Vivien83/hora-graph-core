# 01 — Architecture globale

> Modules, data flow, separation des responsabilites.

---

## Vue d'ensemble

```
┌──────────────────────────────────────────────────────────────────┐
│                        Public API (lib.rs)                       │
│  HoraCore::open() / add_entity() / search() / dream_cycle()     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │   core/   │  │  search/ │  │ memory/  │  │    temporal/     │ │
│  │          │  │          │  │          │  │                  │ │
│  │ Entity   │  │ Vector   │  │ ACT-R    │  │ Bi-temporal      │ │
│  │ Edge     │  │ BM25     │  │ CLS      │  │ Timeline         │ │
│  │ Episode  │  │ HNSW     │  │ Dark     │  │ Facts-at         │ │
│  │ Dedup    │  │ Hybrid   │  │ Reconsol │  │                  │ │
│  │ Types    │  │ Traverse │  │          │  │                  │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│       │              │              │                 │           │
│  ─────┴──────────────┴──────────────┴─────────────────┴────────  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    storage/ (trait StorageOps)                │ │
│  │                                                              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │ │
│  │  │ Embedded │  │  SQLite  │  │ Postgres │  │   Memory    │  │ │
│  │  │ (.hora)  │  │ (FTS5)   │  │(pgvector)│  │  (tests)    │  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                          ffi/                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ napi-rs  │  │  PyO3    │  │  WASM    │  │     C FFI       │ │
│  │ (Node)   │  │ (Python) │  │(browser) │  │   (header.h)    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Modules et responsabilites

### `core/` — Structures de donnees fondamentales

| Fichier | Responsabilite | Depend de |
|---------|---------------|-----------|
| `types.rs` | EntityId, EdgeId, Config, enums | rien |
| `entity.rs` | Entity struct, EntityStore (B+ tree in-memory) | types |
| `edge.rs` | Edge struct, EdgeStore (CSR adjacency lists) | types |
| `episode.rs` | Episode struct, fast append store | types |
| `dedup.rs` | Deduplication triple (nom + Jaccard + cosine) | entity, search/vector |

**Invariants :**
- EntityId est un u64 auto-increment, jamais reutilise
- EdgeId est un u64 auto-increment, jamais reutilise
- Un Edge reference toujours deux EntityId valides (ou un dark node)
- L'EntityStore maintient un index name→id pour la dedup

### `search/` — Moteur de recherche hybride

| Fichier | Responsabilite | Depend de |
|---------|---------------|-----------|
| `vector.rs` | Cosine similarity SIMD, brute-force + HNSW | core/types |
| `bm25.rs` | Inverted index, tokenization, TF-IDF scoring | core/types |
| `hybrid.rs` | Reciprocal Rank Fusion (RRF) | vector, bm25 |
| `traversal.rs` | BFS/DFS avec spreading activation | core/entity, core/edge, memory/activation |

**Invariants :**
- vector_search retourne toujours exactement k resultats (ou moins si < k entites)
- hybrid_search ne retourne jamais un dark node (sauf flag explicite)
- Les scores sont normalises [0.0, 1.0]

### `memory/` — Modele bio-inspire

| Fichier | Responsabilite | Depend de |
|---------|---------------|-----------|
| `activation.rs` | ACT-R Base-Level Learning, spreading, fan effect | core/entity, core/edge |
| `consolidation.rs` | Dream cycle, CLS transfer, SHY downscaling | core/episode, core/entity |
| `reconsolidation.rs` | Update-on-access, labile window | activation |
| `dark_nodes.rs` | Seuil de silence, recovery, GC | activation |

**Invariants :**
- Chaque access a une entite met a jour son activation (sauf lecture raw)
- Un dark node n'est jamais supprime physiquement, seulement marque
- Le dream cycle est idempotent (relancer ne change pas le resultat)

### `temporal/` — Modele bi-temporel

| Fichier | Responsabilite | Depend de |
|---------|---------------|-----------|
| `bitemporal.rs` | Index (valid_at, created_at), queries temporelles | core/edge |
| `timeline.rs` | Historique d'une entite, evolution dans le temps | bitemporal |

**Invariants :**
- Chaque fait a deux axes temporels : valid_at (monde reel) et created_at (systeme)
- `invalidate_fact()` set invalid_at, ne supprime jamais
- `facts_at(t)` retourne l'etat du monde a l'instant t

### `storage/` — Backends pluggables

| Fichier | Responsabilite | Depend de |
|---------|---------------|-----------|
| `traits.rs` | Trait `StorageOps` | core/types |
| `embedded/` | Format .hora, WAL, mmap | traits |
| `sqlite.rs` | Backend SQLite via rusqlite | traits |
| `postgres.rs` | Backend PostgreSQL via tokio-postgres | traits |
| `memory.rs` | Backend in-memory pour tests | traits |

**Invariants :**
- Tout backend implemente exactement le meme trait
- Le switch de backend ne change pas le comportement observable
- Embedded est le backend par defaut, les autres sont des feature flags

### `ffi/` — Bindings multi-langages

| Fichier | Responsabilite | Depend de |
|---------|---------------|-----------|
| `napi.rs` | Bindings Node.js via napi-rs | tout |
| `pyo3.rs` | Bindings Python via PyO3 | tout |
| `wasm.rs` | Bindings WASM via wasm-bindgen | tout (sauf mmap, postgres) |

---

## Data flow — Operations principales

### Add Entity
```
API: hora.add_entity(type, name, props)
  │
  ├─ dedup.check(name, embedding?)
  │   ├─ nom exact match? → return existing ID
  │   ├─ Jaccard > 0.85? → return existing ID
  │   └─ cosine > 0.92? → return existing ID
  │
  ├─ entity_store.insert(entity)
  ├─ string_pool.insert(name)
  ├─ vector_store.insert(embedding?)  // si fourni
  ├─ bm25_index.insert(name + props)
  ├─ activation.init(entity_id, now)
  │
  └─ storage.flush_if_needed()
```

### Search (hybrid)
```
API: hora.search(query_text, opts)
  │
  ├─ vector_search(query_embedding, k*2)
  │   └─ SIMD cosine brute-force ou HNSW
  │
  ├─ bm25_search(query_text, k*2)
  │   └─ tokenize → lookup posting lists → score
  │
  ├─ hybrid.rrf_fusion(vec_results, bm25_results, k)
  │   └─ RRF: score = Σ 1/(rrf_k + rank_i)
  │
  ├─ activation.boost(results)
  │   └─ score_final = rrf_score * activation_factor
  │
  ├─ filter: exclure dark nodes (sauf opts.include_dark)
  ├─ filter: appliquer temporal filter si opts.at_time
  │
  ├─ activation.record_access(result_ids)  // side-effect
  │
  └─ return top-k results
```

### Dream Cycle
```
API: hora.dream_cycle()
  │
  ├─ 1. SHY downscaling
  │   └─ ∀ entity: activation *= downscaling_factor (0.75-0.80)
  │
  ├─ 2. Interleaved replay
  │   └─ Select mix ancien + recent episodes
  │   └─ Re-activate les entites mentionnees
  │
  ├─ 3. CLS transfer
  │   └─ Episodes vus 3+ fois → creer/renforcer fait semantique
  │   └─ Decontextualiser (retirer source_ref episodique)
  │
  ├─ 4. Memory linking
  │   └─ Entites co-creees dans fenetre 6h → creer edge temporel
  │
  ├─ 5. Dark node check
  │   └─ ∀ entity: if activation < dark_threshold → mark dark
  │
  └─ 6. GC optionnel
      └─ Dark nodes > 90 jours sans access → candidate for delete
```

---

## Concurrency model

> Detail dans `08-concurrency.md`

**Modele : Single-Writer / Multi-Reader (comme SQLite)**

```
┌─────────────────────────────────────────┐
│              HoraCore                    │
│                                          │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │  RwLock<     │  │  Read-only view  │ │
│  │  WritableDB  │  │  (snapshot)      │ │
│  │  >           │  │                  │ │
│  └──────────────┘  └──────────────────┘ │
│        │ write lock        │ no lock    │
│        ▼                   ▼            │
│  ┌──────────────────────────────────────┐│
│  │         Storage Backend              ││
│  │  WAL for writes, mmap for reads     ││
│  └──────────────────────────────────────┘│
└──────────────────────────────────────────┘

Lecteurs : illimites, sans lock, sur snapshot
Ecrivain : un seul a la fois, via write lock
WAL : les ecritures vont dans le WAL, les lectures voient le dernier checkpoint
```

---

## Error strategy

> Detail dans `09-error-handling.md`

```rust
pub enum HoraError {
    // Storage
    IoError(std::io::Error),
    CorruptedFile { page: u32, expected_checksum: u32, actual: u32 },
    WalCorrupted,

    // Schema
    EntityNotFound(EntityId),
    EdgeNotFound(EdgeId),
    DimensionMismatch { expected: usize, got: usize },
    InvalidEntityType(String),
    InvalidRelationType(String),

    // Capacity
    StringTooLong { max: usize, got: usize },
    TooManyProperties { max: usize, got: usize },
    StorageFull,

    // Concurrency
    WriteLockTimeout,
    TransactionConflict,
}
```

---

## Feature flags (Cargo.toml)

```toml
[features]
default = ["embedded"]

# Storage backends
embedded = []           # Format .hora natif (toujours disponible)
sqlite = ["dep:rusqlite"]
postgres = ["dep:tokio-postgres", "dep:tokio"]

# Search
hnsw = []               # HNSW index (sinon brute-force SIMD)

# Bindings (build-time only)
napi = ["dep:napi", "dep:napi-derive"]
pyo3 = ["dep:pyo3"]
wasm = ["dep:wasm-bindgen"]

# Optional embedding
embedder = ["dep:ort"]  # ONNX Runtime pour embedding integre

# Dev
bench = []
```

**Note :** "zero dependency" signifie zero dep runtime pour le feature set `default`.
Les backends alternatifs (sqlite, postgres) et bindings sont des feature flags opt-in.

---

*Document cree le 2026-03-02. Fait partie de la preparation hora-graph-core.*
