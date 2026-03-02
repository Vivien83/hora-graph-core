# hora-graph-core — Index de la preparation

> Le SQLite du knowledge graph pour l'ere des agents IA.
> Ce dossier contient l'ensemble de la preparation avant de commencer le developpement.

---

## Documents fondamentaux

| # | Document | Description |
|---|----------|-------------|
| 00 | [Vision](00-vision.md) | Positionnement, concurrents, audience, metriques |
| 01 | [Architecture](01-architecture.md) | Modules, data flow, feature flags |
| 02 | [Data Layout](02-data-layout.md) | Structures binaires byte-level, alignement |
| 03 | [Storage Engine](03-storage-engine.md) | Format .hora, pages, WAL, mmap, B+ tree |

## Documents techniques

| # | Document | Description |
|---|----------|-------------|
| 04 | [Bio-Inspired](04-bio-inspired.md) | ACT-R, spreading, dark nodes, dream cycle, FSRS |
| 05 | [Search Engine](05-search-engine.md) | SIMD cosine, BM25, HNSW, RRF hybrid |
| 06 | [Temporal](06-temporal.md) | Modele bi-temporel, valid_at/invalid_at, timeline |
| 07 | [API Design](07-api-design.md) | API Rust + Node.js + Python + WASM |
| 08 | [Concurrency](08-concurrency.md) | Single-Writer / Multi-Reader, RwLock |
| 09 | [Error Handling](09-error-handling.md) | HoraError enum, CRC32, error mapping |

## Qualite & Strategie

| # | Document | Description |
|---|----------|-------------|
| 10 | [Testing Strategy](10-testing-strategy.md) | Pyramide tests, proptest, insta, CI |
| 11 | [Benchmarks](11-benchmarks.md) | Criterion, targets, data generators |
| 12 | [Dependencies](12-dependencies.md) | Zero-dep strategy, ce qu'on reimplemente |
| 13 | [Risks](13-risks.md) | 14 risques identifies avec mitigations |
| 14 | [Migration](14-migration.md) | Evolution format, import/export, CLI |

## Phases de developpement

| Phase | Document | Scope | Duree estimee |
|-------|----------|-------|---------------|
| v0.1 | [Foundations](phases/v0.1-foundations.md) | Types, CRUD, traversal, storage basique, napi-rs | 10-14 jours |
| v0.2 | [Perception](phases/v0.2-perception.md) | Vector SIMD, BM25, hybrid RRF, dedup | 6-10 jours |
| v0.3 | [Memory](phases/v0.3-memory.md) | ACT-R BLL, spreading, dark nodes, FSRS | 7-12 jours |
| v0.4 | [Consolidation](phases/v0.4-consolidation.md) | Dream cycle, CLS, memory linking | 8-13 jours |
| v0.5 | [Robustness](phases/v0.5-robustness.md) | Page-based storage, WAL, mmap, transactions | 14-21 jours |
| v0.6 | [Storage Universal](phases/v0.6-storage-universal.md) | SQLite + PostgreSQL backends | 7-10 jours |
| v0.7 | [Language Universal](phases/v0.7-language-universal.md) | Python, WASM, C FFI, distribution | 9-12 jours |
| v1.0 | [Release](phases/v1.0-release.md) | API freeze, docs, security, publishing | 7-10 jours |

**Total estime : 68-102 jours de developpement**

## Research

| Document | Sujet |
|----------|-------|
| [ACT-R Implementation](research/act-r-implementation.md) | BLL, Petrov, spreading, FSRS, reconsolidation |
| [SIMD Cross-Platform](research/simd-cross-platform.md) | std::arch, AVX2, NEON, WASM SIMD, dispatch |
| [Page-Based Format](research/page-based-format.md) | SQLite reference, WAL, mmap, ACID |
| [HNSW from Scratch](research/hnsw-from-scratch.md) | Algorithme, parametres, persistence, alternatives |
| [napi-rs Patterns](research/napi-rs-patterns.md) | v3, Float32Array, async, distribution npm |
| [Embedding Strategy](research/embedding-strategy.md) | External-first, BM25 zero-dep, ONNX optionnel |

## Plan original

| Document | Description |
|----------|-------------|
| [Plan Original](plan-original.md) | Le plan initial complet (572 lignes) |

---

## Ordre de lecture recommande

1. **Vision** (00) — comprendre le pourquoi
2. **Architecture** (01) — comprendre le comment
3. **Phase v0.1** — comprendre le premier jalon
4. **Data Layout** (02) + **Storage** (03) — les fondations techniques
5. Le reste selon les besoins

---

*Index cree le 2026-03-02.*
