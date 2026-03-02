# 12 — Analyse Dependencies

> Zero-dep = zero dep RUNTIME pour le feature set `default`.
> Certaines deps build-time et optionnelles sont acceptables.

---

## Strategie : "zero runtime dependency"

| Categorie | Policy | Exemples |
|-----------|--------|----------|
| **Runtime core** | ZERO dep | Serde, tokio, thiserror = NON |
| **Build-time/proc-macro** | Accepte si justifie | napi-derive, pyo3-macros |
| **Optional feature** | Accepte sous feature flag | rusqlite, tokio-postgres, ort |
| **Dev/test** | Libre | criterion, proptest, insta |
| **SIMD** | stdlib uniquement | std::arch (stable) |

---

## Dependencies acceptees

### Core (feature = "default")

| Crate | Pourquoi | Alternative zero-dep |
|-------|----------|---------------------|
| **Aucun** | Zero dep runtime | — |

L'error handling utilise `std::error::Error` manuel (pas thiserror).
La serialisation binaire est manuelle (pas serde).
Le RNG utilise un LCG maison ou `std::hash::DefaultHasher`.

### Optionnel — HNSW (feature = "hnsw")

| Crate | Pourquoi | Taille |
|-------|----------|--------|
| `ordered-float` | Ord sur f32 pour BinaryHeap | 15 KB |
| `smallvec` | Neighbor lists inline | 20 KB |

Alternative : implementer OrderedFloat et SmallVec en interne (~200 lignes total). A evaluer au benchmark.

### Optionnel — Storage backends

| Feature | Crate | Justification |
|---------|-------|--------------|
| `sqlite` | `rusqlite` | Backend SQLite, FTS5 |
| `postgres` | `tokio-postgres`, `tokio` | Backend PostgreSQL |

### Optionnel — Bindings

| Feature | Crate | Justification |
|---------|-------|--------------|
| `napi` | `napi`, `napi-derive` | Node.js binding (v3) |
| `pyo3` | `pyo3` | Python binding |
| `wasm` | `wasm-bindgen` | WASM binding |

### Optionnel — Embedding

| Feature | Crate | Justification |
|---------|-------|--------------|
| `embedder` | `ort`, `tokenizers` | ONNX Runtime pour embedding local |

### Dev / Test

| Crate | Usage |
|-------|-------|
| `criterion` | Benchmarks |
| `proptest` | Property-based testing |
| `insta` | Snapshot testing |
| `tempfile` | Fichiers temporaires pour tests |

---

## Ce qu'on reimplemente plutot que dependre

| Fonctionnalite | Crate evite | Notre impl | Lignes |
|----------------|------------|------------|--------|
| Error types | thiserror | `impl std::error::Error` | ~80 |
| Binary serde | serde/bincode | `to_bytes()/from_bytes()` | ~200 |
| CRC32 | crc32fast | Table lookup impl | ~50 |
| VByte encoding | — | Encode/decode u32 | ~20 |
| Tokenization | unicode-segmentation | `char::is_alphanumeric()` | ~30 |
| Stop words | — | Static array hardcode | ~10 |
| mmap wrapper | memmap2 | `libc::mmap` direct | ~100 |
| RNG (level gen) | rand | LCG ou Xoshiro256 | ~30 |
| OrderedFloat | ordered-float | Newtype + Ord impl | ~30 |
| SmallVec | smallvec | ArrayVec maison | ~100 |

**Total : ~650 lignes** pour remplacer toutes les deps optionnelles.

**Decision :** Commencer avec les crates (`ordered-float`, `smallvec`) pour la v0.1, puis evaluer si le remplacement maison vaut le cout en v0.5 (quand la stabilite est prioritaire).

---

*Document cree le 2026-03-02.*
