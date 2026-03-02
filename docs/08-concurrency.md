# 08 — Concurrency Model

> Single-Writer / Multi-Reader, comme SQLite.

---

## Modele choisi

```
┌──────────────────────────────────────────┐
│              HoraCore                     │
│                                           │
│  ┌─────────────────────────────────────┐ │
│  │         RwLock<Inner>               │ │
│  │                                     │ │
│  │  Read lock : N readers simultanes   │ │
│  │  Write lock : 1 writer exclusif     │ │
│  └─────────────────────────────────────┘ │
│                    │                      │
│  ┌─────────────────┴───────────────────┐ │
│  │         Storage Backend             │ │
│  │  WAL : writes append-only           │ │
│  │  mmap : reads zero-copy             │ │
│  └─────────────────────────────────────┘ │
└──────────────────────────────────────────┘

Readers : snapshot-isolated via WAL read marks
Writer : serialise via write lock, append to WAL
Checkpoint : copie WAL → fichier principal (bloque writer, pas readers)
```

## Pourquoi pas fully concurrent ?

| Modele | Complexite | Performance reads | Performance writes |
|--------|-----------|-------------------|-------------------|
| Mutex (excl) | Simple | 1 a la fois | 1 a la fois |
| **RwLock (SW/MR)** | Moyen | **N simultanes** | **1 a la fois** |
| Lock-free | Tres complexe | N simultanes | N (mais contention) |
| MVCC | Tres complexe | N (snapshot) | N (mais conflicts) |

Un KG a un ratio reads/writes tres eleve (90%+ reads). SW/MR est optimal.

## Implementation

```rust
pub struct HoraCore {
    inner: Arc<RwLock<HoraInner>>,
}

// Operations read : prennent &self (shared ref)
impl HoraCore {
    pub fn get_entity(&self, id: EntityId) -> Result<Option<Entity>> {
        let inner = self.inner.read().unwrap();
        inner.storage.get_entity(id)
    }

    pub fn search(&self, ...) -> Result<Vec<SearchHit>> {
        let inner = self.inner.read().unwrap();
        // ...
    }
}

// Operations write : prennent &self mais acquierent write lock en interne
impl HoraCore {
    pub fn add_entity(&self, ...) -> Result<EntityId> {
        let mut inner = self.inner.write().unwrap();
        inner.storage.put_entity(entity)?;
        inner.bm25_index.insert(id, &text)?;
        Ok(id)
    }
}
```

## Thread safety du binding Node.js

```rust
#[napi]
pub struct HoraCore {
    inner: Arc<RwLock<HoraInner>>,  // Send + Sync
}

// napi-rs async : les closures capturent Arc, thread-safe
#[napi]
impl HoraCore {
    #[napi]
    pub async fn search(&self, ...) -> Result<Vec<SearchHitJs>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let guard = inner.read().unwrap();
            // ... search logic
        }).await?
    }
}
```

## Limites

- **Pas de transactions multi-statement en v0.1** — chaque op est atomique individuellement
- **Write starvation possible** si beaucoup de readers — mitigable avec fair RwLock (`parking_lot`)
- **Dream cycle prend le write lock** pendant toute sa duree — a executer en off-peak

---

*Document cree le 2026-03-02.*
