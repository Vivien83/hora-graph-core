# hora-graph-core — Plan d'implémentation

> Le SQLite du knowledge graph pour l'ère des agents IA.
> Moteur bio-inspiré, embedded, zero-dependency, Rust.

---

## Vision

Un moteur de knowledge graph qui modélise la mémoire comme le cerveau humain :
- Activation decay (ACT-R)
- Consolidation nocturne (dream cycle)
- Oubli actif (Rac1-inspired)
- Recherche hybride (vector + BM25 + RRF + BFS)
- Bi-temporal natif
- Un fichier. Zéro config. Zéro dépendance.

---

## Principes fondateurs

```
ZERO DEPENDENCY    — stdlib Rust + SIMD intrinsics uniquement
PORTABLE           — compile partout (macOS, Linux, Windows, WASM)
FAST               — SIMD vectorisé, CSR cache-friendly, O(1)/hop
UNIVERSAL          — Node (napi-rs), Python (PyO3), WASM, C FFI
CONFIGURABLE       — storage, dimensions, types, relations, tout pluggable
BIO-INSPIRED       — copie le cerveau, pas Neo4j
```

---

## Positionnement concurrentiel

### Case vide identifiée
Aucune solution ne combine : embedded + temporal + hybrid search + bio-inspired + Rust + TS-native.

### Concurrents analysés

| Solution | Forces | Faiblesses critiques |
|---|---|---|
| **Mem0** | Plug-and-play, 24+ backends | 3 infras externes, pas de temporal, graph = paywall |
| **Zep/Graphiti** | Bi-temporal rigoureux, hybrid search | Neo4j obligatoire, Python-only |
| **Cognee** | Zero-config (Kuzu+LanceDB+SQLite) | Pas de temporal, Python-only |
| **CozoDB** | Rust, embedded, HNSW | Dev ralenti, Datalog = friction |
| **Kuzu** | Embedded, Cypher, performant | OLAP-oriented, pas de temporal |
| **LangMem** | Intégration LangChain | Pas de graphe, purement vectoriel |
| **Microsoft GraphRAG** | Community summaries | $33K indexing, pas incrémental |

---

## Architecture bio-inspirée

### Correspondances cerveau → hora

| Cerveau | Mécanisme | hora-graph-core |
|---|---|---|
| Engram (CREB) | Sélection compétitive 10-30% | Encodage sélectif, dédup agressive |
| Dentate Gyrus | Pattern separation | Embedding orthogonalisation |
| CA3 | Pattern completion (Hopfield) | Graph expansion depuis cue partielle |
| Spreading activation | Propagation avec decay/distance | BFS avec atténuation par hop |
| ACT-R BLL | `B = ln(Σ tⱼ^(-d))`, d≈0.5 | Score d'activation par nœud/edge |
| Fan Effect | Plus d'associations = dilution | `Sⱼᵢ = S - ln(fan)` |
| Rac1 pathway | Oubli actif moléculaire | Dark nodes (silencieux, récupérables) |
| CLS fast/slow | Hippocampe/néocortex | Épisodique (fast write) / sémantique (consolidé) |
| Reconsolidation | Accès déstabilise 4-6h | Update window après chaque access |
| Memory linking | 6h co-allocation window | Liens temporels auto entre co-créés |
| SHY (sommeil) | Downscaling homéostatique | Dream cycle : normalisation périodique |
| Sharp-wave ripples | Replay compressé pendant sommeil | Consolidation interleaved offline |
| Amygdale | Salience émotionnelle | emotional_weight sur activation |
| Working memory | 4±1 chunks | Context window management |
| Épisodique→sémantique | Décontextualisation progressive | Transformation par répétition |
| FSRS | `R = (1 + t/(S·w₂₀))^(-1)` | Scheduling de renforcement optimal |

### Formules clés

```
// ACT-R Base-Level Learning
B(i) = ln(Σⱼ tⱼ^(-d))           // d = 0.5, tⱼ = temps depuis accès j

// ACT-R Total Activation
A(i) = B(i) + Σⱼ(Wⱼ · Sⱼᵢ) + ε  // spreading + noise

// Spreading Activation
Sⱼᵢ = S - ln(fan_i)              // fan effect

// Retrieval Probability
P(i) = 1 / (1 + e^(-(A(i) - τ)/s))  // sigmoid, τ = threshold

// FSRS Retrievability
R = (1 + t / (S · w₂₀))^(-1)     // power-law decay

// Cosine Similarity (SIMD)
cos(a,b) = dot(a,b) / (|a| · |b|)  // AVX2: 8 f32/instruction
```

---

## Data Layout

### Entity (46 bytes)
```rust
struct Entity {
    id: u64,                    // auto-increment (8B)
    entity_type: u8,            // 256 types max (1B)
    name_offset: u32,           // string pool pointer (4B)
    name_len: u16,              // name length (2B)
    properties_offset: u32,     // property page pointer (4B)
    embedding_offset: u32,      // vector page pointer, 0=none (4B)
    created_at: i64,            // unix millis (8B)
    last_seen: i64,             // unix millis (8B)
    adjacency_offset: u32,      // CSR edge list pointer (4B)
    adjacency_count: u16,       // outgoing edge count (2B)
    activation: f32,            // ACT-R activation score (4B) — AJOUT BIO
}
// 1M entités ≈ 48 MB
```

### Edge (38 bytes)
```rust
struct Edge {
    target: u64,                // entity ID cible (8B)
    relation: u8,               // relation type (1B)
    confidence: u8,             // 0-255 → 0.0-1.0 (1B)
    valid_at: i64,              // bi-temporal: monde réel (8B)
    invalid_at: i64,            // 0 = toujours valide (8B)
    created_at: i64,            // bi-temporal: système (8B) — AJOUT
    description_offset: u32,    // string pool (4B)
    metadata_offset: u32,       // metadata page (4B)
}
```

### Fichier .hora (page-based)
```
┌─────────────────────────────────────────┐
│  Header (magic "HORA", version, config) │
│  Page directory                         │
│  Entity pages (B+ tree sur ID)          │
│  Edge pages (CSR blocks contigus)       │
│  Property pages (columnar)              │
│  String pool (noms, descriptions)       │
│  Vector pages (alignés 32B pour AVX2)   │
│  Temporal index pages                   │
│  BM25 inverted index (VByte posting)    │
│  Activation log pages                   │
│  Episode pages (fast store)             │
│  Freelist                               │
│  WAL (write-ahead log)                  │
└─────────────────────────────────────────┘
Page size: 4KB (configurable)
```

---

## Storage Pluggable (enum dispatch)

```rust
pub enum StorageBackend {
    Embedded(EmbeddedStorage),   // fichier .hora, WAL, mmap
    Sqlite(SqliteStorage),       // rusqlite, FTS5
    Postgres(PostgresStorage),   // tokio-postgres, pgvector
    Memory(MemoryStorage),       // tests, éphémère
}

pub trait StorageOps {
    // CRUD
    fn put_entities(&mut self, batch: &[Entity]) -> Result<()>;
    fn get_entity(&self, id: EntityId) -> Result<Option<Entity>>;
    fn put_edges(&mut self, batch: &[Edge]) -> Result<()>;
    fn scan_edges(&self, source: EntityId) -> Result<EdgeIterator>;
    fn delete_entity(&mut self, id: EntityId) -> Result<()>;
    fn delete_edge(&mut self, id: EdgeId) -> Result<()>;

    // Search
    fn vector_search(&self, query: &[f32], k: usize, filter: Option<Filter>) -> Result<Vec<SearchHit>>;
    fn text_search(&self, query: &str, k: usize) -> Result<Vec<SearchHit>>;

    // Temporal
    fn facts_at(&self, timestamp: i64) -> Result<Vec<Edge>>;
    fn entity_history(&self, id: EntityId) -> Result<Vec<Edge>>;

    // Lifecycle
    fn flush(&mut self) -> Result<()>;
    fn snapshot(&self, path: &Path) -> Result<()>;
    fn compact(&mut self) -> Result<()>;
}
```

---

## Configuration (tout paramétrable)

```rust
pub struct HoraConfig {
    // Storage
    pub storage: StorageType,           // Embedded | Sqlite | Postgres | Memory
    pub path: Option<PathBuf>,          // fichier .hora ou connection string

    // Schema (extensible)
    pub embedding_dims: usize,          // 384, 768, 1536... au choix
    pub entity_types: Vec<String>,      // extensible, pas hardcodé
    pub relation_types: Vec<String>,    // extensible
    pub max_properties: usize,          // par entité (défaut: 64)

    // Bio-inspired
    pub activation: ActivationConfig {
        pub decay_param: f64,           // d = 0.5 (ACT-R)
        pub retrieval_threshold: f64,   // τ = 0.0
        pub noise: f64,                 // s = 0.25
        pub emotional_weight: bool,     // amygdale on/off
        pub reconsolidation_window_secs: u64,  // 21600 (6h)
        pub dark_node_threshold: f64,   // activation sous laquelle = dark
    },
    pub consolidation: ConsolidationConfig {
        pub enabled: bool,              // dream cycle on/off
        pub interval_secs: u64,         // 86400 (24h)
        pub episodic_to_semantic: bool, // CLS transfer
        pub downscaling_factor: f64,    // SHY: 0.75-0.80
        pub max_replay_items: usize,    // par cycle
    },

    // Search
    pub search: SearchConfig {
        pub semantic_weight: f64,       // RRF: 0.7
        pub bm25_weight: f64,           // RRF: 0.3
        pub rrf_k: usize,              // 60
        pub default_top_k: usize,       // 10
        pub hnsw_enabled: bool,         // false sous 100K vecs
        pub hnsw_m: usize,             // 16
        pub hnsw_ef_construction: usize, // 200
    },

    // Deduplication
    pub dedup: DedupConfig {
        pub name_normalization: bool,   // lowercase + trim
        pub jaccard_threshold: f64,     // 0.85
        pub cosine_threshold: f64,      // 0.92
    },

    // Performance
    pub page_size: usize,               // 4096
    pub wal_enabled: bool,              // true
    pub mmap_enabled: bool,             // true (embedded only)
    pub batch_size: usize,              // 500
}
```

---

## API publique

```rust
// Ouverture / création
let hora = HoraCore::open("memory.hora", config)?;  // ou ::new() pour in-memory

// CRUD Entités
let id = hora.add_entity("project", "hora-engine", props)?;
hora.update_entity(id, updates)?;
hora.delete_entity(id)?;
let entity = hora.get_entity(id)?;

// CRUD Facts (edges)
let fact_id = hora.add_fact(source_id, target_id, "depends_on", "hora depends on Rust", meta)?;
hora.update_fact(fact_id, updates)?;
hora.delete_fact(fact_id)?;
hora.invalidate_fact(fact_id)?;  // bi-temporal: marque invalid_at, ne supprime pas

// Search
let results = hora.search("how does auth work?", SearchOpts::default())?;
let results = hora.vector_search(&embedding, 10)?;
let results = hora.text_search("authentication", 10)?;

// Graph Traversal
let subgraph = hora.traverse(start_id, depth: 3)?;
let timeline = hora.timeline(entity_id)?;
let facts_then = hora.facts_at(timestamp)?;

// Bio-inspired
hora.record_access(entity_id)?;  // met à jour activation + reconsolidation
let activation = hora.get_activation(entity_id)?;
hora.dream_cycle()?;             // consolidation manuelle
let dark = hora.dark_nodes()?;   // nœuds silencieux

// Episodes (fast store)
hora.add_episode(source_type, source_ref, entities, facts)?;

// Persistence
hora.flush()?;
hora.snapshot("backup.hora")?;
hora.compact()?;
hora.close()?;

// Stats
let stats = hora.stats()?;  // entities, facts, episodes, dark_nodes, activation_mean
```

---

## Bindings

### Node.js (napi-rs) — first-class
```typescript
import { HoraCore } from '@hora-engine/graph-core';

const hora = HoraCore.open('memory.hora', { embeddingDims: 384 });
const id = hora.addEntity('project', 'my-app', { language: 'typescript' });
const results = hora.search('authentication patterns', { topK: 10 });
hora.close();
```

### Python (PyO3)
```python
from hora_graph_core import HoraCore

hora = HoraCore.open("memory.hora", embedding_dims=384)
entity_id = hora.add_entity("project", "my-app", {"language": "python"})
results = hora.search("authentication patterns", top_k=10)
hora.close()
```

### WASM (wasm-bindgen)
```javascript
import init, { HoraCore } from 'hora-graph-core-wasm';
await init();
const hora = HoraCore.new_memory({ embeddingDims: 384 });
// Même API, dans le navigateur
```

---

## Versions (roadmap)

### v0.1 — Fondations (Hippocampe + Synapses)
**Scope :**
- [ ] Scaffolding Rust crate (`hora-graph-core`)
- [ ] Types de base : Entity, Edge, EntityId, EdgeId
- [ ] Entity store : B+ tree en mémoire, CRUD
- [ ] Edge store : CSR adjacency, CRUD
- [ ] String pool : noms et descriptions
- [ ] Config extensible : entity types, relation types
- [ ] Embedded storage : fichier .hora basique (serde binaire, pas encore page-based)
- [ ] Memory storage : pour les tests
- [ ] BFS traversal : `traverse(id, depth)`
- [ ] Timeline : `timeline(entity_id)`
- [ ] Bi-temporal basique : `facts_at(timestamp)`
- [ ] napi-rs binding : Node.js wrapper
- [ ] Tests unitaires complets
- [ ] Benchmark basique (CRUD ops/sec)

**Critères de validation :**
- 1M entities + 5M edges chargés en < 500ms
- BFS 3-hop sur 1M nœuds < 5ms
- CRUD : > 100K ops/sec

### v0.2 — Perception (Dentate Gyrus + Cortex Préfrontal)
**Scope :**
- [ ] Vector store : stockage aligné 32B, brute-force SIMD (AVX2/NEON)
- [ ] Cosine similarity SIMD : 384d en ~48 instructions AVX2
- [ ] BM25 inverted index : posting lists VByte
- [ ] Hybrid search RRF : fusion semantic (0.7) + BM25 (0.3)
- [ ] Déduplication triple : nom + Jaccard + cosine
- [ ] Pattern separation : orthogonalisation check à l'insert
- [ ] Benchmark search (latence, recall@10)

**Critères de validation :**
- Cosine top-100 sur 100K vecs (384d) < 5ms
- BM25 search < 2ms
- Hybrid search < 10ms
- Dedup : 0 duplicats sur jeu de test standardisé

### v0.3 — Mémoire (Amygdale + Modèle Temporel)
**Scope :**
- [ ] ACT-R activation model : `B = ln(Σ tⱼ^(-d))`
- [ ] Spreading activation : propagation avec decay
- [ ] Fan Effect : dilution par nombre d'associations
- [ ] Reconsolidation window : update-on-access (6h)
- [ ] Emotional weight : salience boosting
- [ ] Dark nodes : nœuds sous threshold, exclus du search, récupérables
- [ ] Access recording : chaque query met à jour les activations
- [ ] Temporal index complet : bi-temporal queries optimisées

**Critères de validation :**
- Activation scores matchent les prédictions ACT-R (test avec données publiées)
- Dark nodes correctement exclus du search mais récupérables par ID
- Reconsolidation modifie vs crée selon la fenêtre

### v0.4 — Sommeil (Consolidation CLS)
**Scope :**
- [ ] Episode store : épisodique fast-write
- [ ] Dream cycle : consolidation périodique
- [ ] CLS transfer : épisodique → sémantique par répétition
- [ ] SHY downscaling : normalisation homéostatique des poids
- [ ] Interleaved replay : mélange ancien + nouveau
- [ ] Memory linking : 6h co-allocation window
- [ ] Garbage collection : nettoyage dark nodes anciens

**Critères de validation :**
- Dream cycle transforme épisodes répétés en faits sémantiques
- Downscaling réduit activation globale de ~20% par cycle
- Memory linking crée des edges temporels entre co-créés

### v0.5 — Robustesse (Neurologie)
**Scope :**
- [ ] Page-based file format : 4KB pages, B+ tree on-disk
- [ ] WAL (Write-Ahead Log) : crash recovery
- [ ] mmap : zero-copy read path
- [ ] HNSW index : pour >100K vecteurs
- [ ] Checksums par page : détection corruption
- [ ] Compaction : défragmentation du fichier .hora
- [ ] ACID transactions

**Critères de validation :**
- Crash recovery : kill -9 pendant écriture → données intactes
- HNSW : recall@10 > 0.95, latence < 2ms sur 1M vecs
- Compaction réduit la taille fichier de >30%

### v0.6 — Universalité Storage
**Scope :**
- [ ] SQLite backend : rusqlite + FTS5 pour BM25
- [ ] PostgreSQL backend : tokio-postgres + pgvector
- [ ] Async/sync bridge : spawn_blocking pour backends sync
- [ ] Migration tool : convert entre backends
- [ ] Benchmark comparatif : embedded vs SQLite vs Postgres

**Critères de validation :**
- Même API, même résultats, quel que soit le backend
- Migration embedded↔SQLite↔Postgres sans perte

### v0.7 — Universalité Langages
**Scope :**
- [ ] PyO3 bindings : Python wrapper
- [ ] wasm-bindgen : compilation WASM
- [ ] C FFI : header C pour intégrations exotiques
- [ ] Cross-platform builds : CI/CD pour macOS ARM/x64, Linux x64, Windows x64
- [ ] npm publish : `@hora-engine/graph-core`
- [ ] PyPI publish : `hora-graph-core`
- [ ] crates.io publish : `hora-graph-core`

**Critères de validation :**
- `npm install @hora-engine/graph-core` fonctionne sur les 3 OS
- `pip install hora-graph-core` fonctionne sur les 3 OS
- WASM fonctionne dans Chrome, Firefox, Safari

### v1.0 — Production Release
**Scope :**
- [ ] Documentation complète (API, guide, exemples)
- [ ] Benchmarks publiés vs Mem0, Zep, CozoDB
- [ ] Fuzzing (cargo-fuzz) pour robustesse
- [ ] Audit sécurité mémoire (miri)
- [ ] Logo, site web, README polished
- [ ] Examples : chatbot memory, RAG pipeline, personal assistant
- [ ] Migration guide depuis hora-engine TS

**Critères de validation :**
- 0 undefined behavior (miri clean)
- 0 memory leaks (valgrind clean)
- Benchmarks documentés et reproductibles
- 3+ examples fonctionnels

---

## Structure du crate

```
hora-graph-core/
├── Cargo.toml
├── src/
│   ├── lib.rs                  // API publique
│   ├── core/
│   │   ├── mod.rs
│   │   ├── entity.rs           // Entity struct + store
│   │   ├── edge.rs             // Edge struct + CSR store
│   │   ├── episode.rs          // Episode fast store
│   │   ├── types.rs            // EntityId, EdgeId, Config
│   │   └── dedup.rs            // Triple deduplication
│   ├── search/
│   │   ├── mod.rs
│   │   ├── vector.rs           // SIMD brute-force + HNSW
│   │   ├── bm25.rs             // Inverted index + VByte
│   │   ├── hybrid.rs           // RRF fusion
│   │   └── traversal.rs        // BFS/DFS spreading activation
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── activation.rs       // ACT-R model
│   │   ├── consolidation.rs    // Dream cycle, CLS
│   │   ├── reconsolidation.rs  // Update-on-access
│   │   └── dark_nodes.rs       // Silent engrams
│   ├── temporal/
│   │   ├── mod.rs
│   │   ├── bitemporal.rs       // Bi-temporal index
│   │   └── timeline.rs         // Entity history
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── traits.rs           // StorageOps trait
│   │   ├── embedded/
│   │   │   ├── mod.rs
│   │   │   ├── file.rs         // Page-based .hora format
│   │   │   ├── wal.rs          // Write-ahead log
│   │   │   └── mmap.rs         // Memory-mapped reads
│   │   ├── sqlite.rs           // SQLite backend
│   │   ├── postgres.rs         // PostgreSQL backend
│   │   └── memory.rs           // In-memory backend
│   └── ffi/
│       ├── napi.rs             // Node.js bindings
│       ├── pyo3.rs             // Python bindings
│       └── wasm.rs             // WASM bindings
├── benches/
│   ├── crud.rs
│   ├── search.rs
│   └── traversal.rs
├── tests/
│   ├── integration/
│   └── fixtures/
└── bindings/
    ├── node/                   // napi-rs package
    │   ├── package.json
    │   ├── index.js
    │   └── index.d.ts
    ├── python/                 // PyO3 package
    │   ├── pyproject.toml
    │   └── hora_graph_core/
    └── wasm/                   // wasm-pack package
        └── pkg/
```

---

## Performance cibles

| Opération | Cible v1.0 | Référence |
|---|---|---|
| Open 1M nœuds (embedded) | < 50ms | SQLite: ~1ms |
| BFS 3-hop sur 1M nœuds | < 1ms | Neo4j: ~5-10ms |
| Cosine top-100 / 100K vecs 384d | < 5ms | Qdrant: ~2ms |
| HNSW top-100 / 1M vecs 384d | < 2ms | |
| BM25 search | < 2ms | |
| Hybrid search (vec + BM25 + RRF) | < 10ms | Current TS: ~50ms |
| Add entity | < 1μs | In-memory + WAL async |
| Batch insert 10K entities | < 10ms | |
| Fichier 100K entities + 500K edges | ~50 MB | Compact binaire |
| Dream cycle 10K nœuds | < 100ms | |
| Memory footprint 100K entities | < 50 MB | |

---

## Risques et mitigations

| Risque | Impact | Mitigation |
|---|---|---|
| Rust learning curve (ownership, lifetimes) | Ralentissement | Arena-based allocation pour le graph, évite les lifetime complexes |
| Cross-platform SIMD | Portabilité | Feature flags : AVX2 (x86), NEON (ARM), fallback scalaire |
| napi-rs breaking changes | Binding cassé | Pin la version, tests CI sur chaque plateforme |
| Scope creep | Deadline | Strict versioning, chaque version = scope fixe |
| CozoDB rattrape | Concurrence | Différenciation bio-inspired, TypeScript-first |
| Performance pas au niveau | Crédibilité | Benchmarks continus dès v0.1 |

---

## Décisions prises

1. **Nom** : `hora-graph-core`
2. **Langage** : Rust, zéro dépendance (stdlib + SIMD)
3. **Storage** : enum dispatch (Embedded, SQLite, Postgres, Memory)
4. **Bio-inspiré** : ACT-R complet, CLS, dark nodes, dream cycle, reconsolidation
5. **Configurable** : tout (dims, types, relations, activation params, storage)
6. **Format** : page-based .hora (inspiré SQLite)
7. **Bindings** : napi-rs (Node) first, puis PyO3 (Python) et WASM
8. **Approche** : incrémentale, v0.1 → v1.0 en 8 versions

---

*Plan créé le 2026-03-01. Prêt pour implémentation sur signal de l'utilisateur.*
