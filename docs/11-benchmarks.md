# 11 — Benchmarks & Performance Targets

> Chaque operation a un budget temps. Mesurer avant d'optimiser.

---

## Methodologie

### Outil : Criterion.rs

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "crud"
harness = false

[[bench]]
name = "search"
harness = false

[[bench]]
name = "traversal"
harness = false

[[bench]]
name = "storage"
harness = false
```

### Environnement de reference

| Spec | Valeur |
|------|--------|
| CPU | Apple M2 (8 cores) ou AMD Zen 4 |
| RAM | 16 GB |
| Disk | NVMe SSD |
| OS | macOS 14+ / Linux 6.x |
| Rust | stable (latest) |
| Profile | `--release` |

---

## Targets par operation

### CRUD (Memory backend)

| Operation | Cible | Mesure |
|-----------|-------|--------|
| `add_entity` (sans embedding) | < 200ns | ops/sec > 5M |
| `add_entity` (avec embedding 384d) | < 500ns | ops/sec > 2M |
| `get_entity` par ID | < 50ns | ops/sec > 20M |
| `update_entity` (name change) | < 200ns | ops/sec > 5M |
| `delete_entity` (sans cascade) | < 100ns | ops/sec > 10M |
| `delete_entity` (avec cascade 10 edges) | < 1us | ops/sec > 1M |
| `add_fact` | < 300ns | ops/sec > 3M |
| Batch insert 100K entities | < 500ms | 200K/sec |

### CRUD (Embedded backend)

| Operation | Cible | Mesure |
|-----------|-------|--------|
| `add_entity` (WAL write) | < 1us | ops/sec > 1M |
| `get_entity` par ID (mmap) | < 500ns | ops/sec > 2M |
| `flush` (1K pending ops) | < 5ms | |
| `checkpoint` (WAL → file) | < 10ms | |
| `open` 1M entities (mmap) | < 50ms | |
| `open` 1M entities (read) | < 500ms | |

### Search

| Operation | Dataset | Cible |
|-----------|---------|-------|
| Cosine brute-force top-100 | 100K vecs, 384d | < 5ms |
| Cosine brute-force top-100 | 1M vecs, 384d | < 50ms |
| Cosine HNSW top-100 | 1M vecs, 384d | < 1ms |
| BM25 search top-10 | 100K docs | < 2ms |
| BM25 search top-10 | 1M docs | < 10ms |
| Hybrid (vec + BM25 + RRF) | 100K | < 10ms |
| Hybrid (vec + BM25 + RRF) | 1M | < 60ms |

### Graph Traversal

| Operation | Dataset | Cible |
|-----------|---------|-------|
| BFS 1-hop | 100K nodes, 500K edges | < 100us |
| BFS 2-hop | 100K nodes, 500K edges | < 1ms |
| BFS 3-hop | 100K nodes, 500K edges | < 5ms |
| Timeline (1 entity, 100 facts) | Any | < 50us |
| `facts_at` | 100K edges | < 5ms |

### Memory Model

| Operation | Cible |
|-----------|-------|
| `compute_activation` (1 entity) | < 100ns |
| `record_access` | < 50ns |
| `spread_activation` (depth=3) | < 1ms |
| `dream_cycle` (10K entities) | < 100ms |
| `dream_cycle` (100K entities) | < 1s |
| `dark_node_pass` (100K entities) | < 50ms |

### Storage Lifecycle

| Operation | Cible |
|-----------|-------|
| Create new .hora file | < 1ms |
| `snapshot` (10MB file) | < 50ms |
| `compact` (30% fragmentation) | < 500ms |
| WAL recovery (1000 frames) | < 100ms |

---

## Benchmarks Criterion — Structure

### `benches/crud.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_add_entity(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_entity");

    group.bench_function("no_embedding", |b| {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        b.iter(|| {
            hora.add_entity("test", "entity", None, None).unwrap();
        });
    });

    group.bench_function("with_embedding_384", |b| {
        let mut hora = HoraCore::new(HoraConfig { embedding_dims: 384, ..default() }).unwrap();
        let embedding = vec![0.1f32; 384];
        b.iter(|| {
            hora.add_entity("test", "entity", None, Some(&embedding)).unwrap();
        });
    });

    group.finish();
}

fn bench_get_entity(c: &mut Criterion) {
    let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
    let ids: Vec<_> = (0..10_000)
        .map(|i| hora.add_entity("test", &format!("e{}", i), None, None).unwrap())
        .collect();

    c.bench_function("get_entity", |b| {
        let mut idx = 0;
        b.iter(|| {
            let id = ids[idx % ids.len()];
            hora.get_entity(id).unwrap();
            idx += 1;
        });
    });
}

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");

    for size in [1_000, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
                for i in 0..size {
                    hora.add_entity("test", &format!("e{}", i), None, None).unwrap();
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_add_entity, bench_get_entity, bench_batch_insert);
criterion_main!(benches);
```

### `benches/search.rs`

```rust
fn bench_cosine_brute_force(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_brute_force");

    for n in [10_000, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut hora = setup_hora_with_vectors(n, 384);
            let query = random_embedding(384);
            b.iter(|| {
                hora.vector_search(&query, 100).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_bm25(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25");

    for n in [10_000, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let hora = setup_hora_with_text(n);
            b.iter(|| {
                hora.text_search("authentication patterns", 10).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_hybrid_search(c: &mut Criterion) {
    let hora = setup_hora_full(100_000, 384);
    let query_embedding = random_embedding(384);

    c.bench_function("hybrid_100k", |b| {
        b.iter(|| {
            hora.search(
                Some("authentication"),
                Some(&query_embedding),
                SearchOpts { top_k: 10, ..default() },
            ).unwrap();
        });
    });
}
```

### `benches/traversal.rs`

```rust
fn bench_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfs");
    let hora = setup_random_graph(100_000, 500_000);
    let start_id = EntityId(1);

    for depth in [1, 2, 3] {
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            b.iter(|| {
                hora.traverse(start_id, TraverseOpts { depth, ..default() }).unwrap();
            });
        });
    }

    group.finish();
}
```

---

## Data generators

```rust
/// Genere un graphe aleatoire pour les benchmarks
pub fn setup_random_graph(n_entities: usize, n_edges: usize) -> HoraCore {
    let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
    let mut rng = SimpleRng::new(42); // seed fixe = reproductible

    // Entities
    let ids: Vec<EntityId> = (0..n_entities)
        .map(|i| hora.add_entity("node", &format!("n{}", i), None, None).unwrap())
        .collect();

    // Edges (random source → random target)
    for _ in 0..n_edges {
        let src = ids[rng.next_usize() % ids.len()];
        let tgt = ids[rng.next_usize() % ids.len()];
        if src != tgt {
            let _ = hora.add_fact(src, tgt, "related", "", None);
        }
    }

    hora
}

/// RNG simple pour benchmarks (pas de dep rand)
struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn next_usize(&mut self) -> usize { self.next_u64() as usize }
}
```

---

## Comparaison concurrents (cibles)

| Benchmark | hora (cible) | Neo4j | Kuzu | CozoDB |
|-----------|-------------|-------|------|--------|
| Insert 1M entities | < 5s | ~30s | ~3s | ~5s |
| Get by ID | < 50ns | ~100us | ~1us | ~5us |
| BFS 3-hop (100K) | < 5ms | ~50ms | ~10ms | ~20ms |
| Cosine top-100 (100K) | < 5ms | N/A | N/A | ~10ms |
| Open 1M entities | < 50ms | ~5s | ~500ms | ~200ms |

Sources : benchmarks publies, extrapolation.

---

## Regression tracking

### Strategie

1. Lancer `cargo bench` a chaque PR
2. Comparer avec la baseline du main
3. Alerter si regression > 20%
4. Stocker les resultats dans `benches/results/`

### Format

```
benches/results/
  YYYY-MM-DD_commit.json
  baseline.json  → derniere release
```

---

*Document cree le 2026-03-02.*
