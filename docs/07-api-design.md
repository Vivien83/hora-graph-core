# 07 — API Design

> API publique Rust + bindings Node/Python/WASM.
> 3 lignes pour ouvrir + ajouter + chercher.

---

## API Rust

```rust
// Ouverture / creation
let hora = HoraCore::open("memory.hora", HoraConfig::default())?;
let hora = HoraCore::new(HoraConfig { storage: StorageType::Memory, ..default() })?;

// --- CRUD Entites ---
let id = hora.add_entity("project", "hora-engine",
    Some(props! { "language" => "Rust", "stars" => 42 }),
    Some(&embedding))?;
hora.update_entity(id, EntityUpdate { name: Some("hora-graph-core"), ..default() })?;
hora.update_entity_embedding(id, &new_embedding)?;
hora.delete_entity(id)?;
let entity = hora.get_entity(id)?;

// --- CRUD Facts (edges) ---
let fact_id = hora.add_fact(source_id, target_id, "depends_on",
    "hora depends on Rust", None)?;
hora.update_fact(fact_id, FactUpdate { confidence: Some(0.95), ..default() })?;
hora.invalidate_fact(fact_id)?;  // bi-temporal: set invalid_at, ne delete pas
hora.delete_fact(fact_id)?;      // suppression physique

// --- Search ---
let results = hora.search(Some("how does auth work?"), Some(&embedding),
    SearchOpts { top_k: 10, ..default() })?;
let results = hora.vector_search(&embedding, 10)?;
let results = hora.text_search("authentication", 10)?;

// --- Graph Traversal ---
let subgraph = hora.traverse(start_id, TraverseOpts { depth: 3, ..default() })?;
let timeline = hora.timeline(entity_id)?;
let facts_then = hora.facts_at(timestamp)?;

// --- Bio-inspired ---
hora.record_access(entity_id)?;
let activation = hora.get_activation(entity_id)?;
hora.dream_cycle()?;
let dark = hora.dark_nodes()?;

// --- Episodes ---
hora.add_episode(EpisodeSource::Conversation, "session-123",
    &[entity_id_1, entity_id_2], &[fact_id_1])?;

// --- Persistence ---
hora.flush()?;
hora.snapshot("backup.hora")?;
hora.compact()?;
hora.close()?;

// --- Stats ---
let stats = hora.stats()?;
// → { entities: 1234, facts: 5678, episodes: 90, dark_nodes: 45, activation_mean: 1.2 }
```

---

## Binding Node.js (napi-rs v3)

```typescript
import { HoraCore } from '@hora-engine/graph-core';

// 3 lignes : ouvrir + ajouter + chercher
const hora = HoraCore.open('memory.hora', { embeddingDims: 384 });
const id = hora.addEntity('project', 'my-app', { language: 'typescript' });
const results = hora.search('authentication patterns', null, { topK: 10 });

// Avec embeddings (Float32Array zero-copy)
const embedding = new Float32Array(384);  // du provider
hora.addEntity('concept', 'auth', { domain: 'security' }, embedding);
hora.vectorSearch(embedding, 10);

// Async pour les operations longues
await hora.dreamCycle();
await hora.compact();

hora.close();
```

### Patterns napi-rs

- `#[napi(object)]` pour les configs/DTOs (plain JS objects)
- `#[napi]` sur struct+impl pour HoraCore (classe JS stateful)
- `Float32Array` pour les embeddings (zero-copy)
- `Result<T>` → JS Error automatique
- Types TS generes automatiquement

### Distribution npm

```
@hora-engine/graph-core           # package principal
@hora-engine/graph-core-darwin-arm64    # macOS Apple Silicon
@hora-engine/graph-core-darwin-x64      # macOS Intel
@hora-engine/graph-core-linux-x64-gnu   # Linux x64
@hora-engine/graph-core-linux-arm64-gnu # Linux ARM64
@hora-engine/graph-core-win32-x64-msvc  # Windows x64
```

Via `optionalDependencies` (pattern SWC/Turbopack).

---

## Binding Python (PyO3)

```python
from hora_graph_core import HoraCore

hora = HoraCore.open("memory.hora", embedding_dims=384)
entity_id = hora.add_entity("project", "my-app", {"language": "python"})
results = hora.search("authentication patterns", top_k=10)
hora.close()
```

---

## Binding WASM

```javascript
import init, { HoraCore } from 'hora-graph-core-wasm';
await init();
const hora = HoraCore.newMemory({ embeddingDims: 384 });
// Meme API, dans le navigateur (pas de mmap, pas de Postgres)
```

---

## Error types exposes

```typescript
// TypeScript
type HoraError =
  | { type: 'DimensionMismatch'; expected: number; got: number }
  | { type: 'EntityNotFound'; id: bigint }
  | { type: 'EdgeNotFound'; id: bigint }
  | { type: 'StorageFull' }
  | { type: 'CorruptedFile'; page: number }
  | { type: 'IoError'; message: string };
```

---

*Document cree le 2026-03-02.*
