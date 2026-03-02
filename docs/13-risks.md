# 13 — Risques identifies & Mitigations

> Identifier avant de coder. Chaque risque a une mitigation concrete.

---

## Matrice des risques

### Legende

| Impact | Description |
|--------|-------------|
| **Critique** | Bloque le projet ou compromet l'architecture |
| **Majeur** | Retard significatif ou degradation de performance |
| **Mineur** | Contournable, impact limite |

| Probabilite | Description |
|-------------|-------------|
| **Haute** | Presque certain (>70%) |
| **Moyenne** | Possible (30-70%) |
| **Basse** | Peu probable (<30%) |

---

## Risques techniques

### R1 — HashMap pas assez rapide pour >1M entities

| | |
|---|---|
| **Phase** | v0.1 |
| **Impact** | Majeur |
| **Probabilite** | Moyenne |
| **Description** | HashMap<EntityId, Entity> a une mauvaise localite cache pour des scans sequentiels. Les rehash a grande taille causent des latency spikes. |
| **Mitigation** | 1. Benchmark des la v0.1f. 2. Si probleme → switcher vers BTreeMap (sequentiel) ou arena allocation (pre-alloue). 3. En v0.5, le B+ tree page-based resout definitivement. |
| **Detection** | Benchmark insert 1M entities : si > 5s = probleme |

### R2 — Serialisation binaire manuelle = bugs de corruption

| | |
|---|---|
| **Phase** | v0.1d |
| **Impact** | Critique |
| **Probabilite** | Haute |
| **Description** | Ecrire to_bytes()/from_bytes() manuellement est error-prone. Off-by-one, endianness, padding = corruption silencieuse. |
| **Mitigation** | 1. proptest : `from_bytes(to_bytes(x)) == x` pour toute structure. 2. CRC32 sur chaque page. 3. Fuzzing avec des fichiers invalides. 4. Magic + version dans le header pour detecter les fichiers non-hora. |
| **Detection** | Tests proptest avec 10K iterations + fuzzing CI |

### R3 — SIMD dispatch runtime : overhead ou incorrect

| | |
|---|---|
| **Phase** | v0.2a |
| **Impact** | Majeur |
| **Probabilite** | Basse |
| **Description** | Le dispatch runtime via OnceLock + function pointer pourrait ajouter overhead ou selectionner le mauvais path. |
| **Mitigation** | 1. OnceLock initialise une seule fois → overhead = 1 indirection. 2. Tests croisees : AVX2 vs NEON vs scalar doivent donner les memes resultats (1e-5). 3. Fallback scalar toujours disponible. |
| **Detection** | Benchmark cosine : si scalar et SIMD < 10% diff = dispatch OK |

### R4 — BM25 tokenizer trop simpliste

| | |
|---|---|
| **Phase** | v0.2b |
| **Impact** | Mineur |
| **Probabilite** | Moyenne |
| **Description** | Le tokenizer `char::is_alphanumeric()` ne gere pas le CamelCase, les acronymes, le chinois/japonais. |
| **Mitigation** | 1. Suffisant pour v0.1-v0.4 (anglais + langues latines). 2. En v0.5, ajouter un tokenizer pluggable via trait. 3. Le hybrid search compense : le vector search capture la semantique que BM25 rate. |
| **Detection** | Tests avec corpus multilingue en v0.5 |

### R5 — HNSW from scratch : qualite de recall insuffisante

| | |
|---|---|
| **Phase** | v0.2 (feature hnsw) |
| **Impact** | Majeur |
| **Probabilite** | Moyenne |
| **Description** | HNSW est un algorithme complexe. Une implementation naive peut avoir un recall@100 mediocre (<85%). |
| **Mitigation** | 1. Commencer par brute-force SIMD (correct par definition). 2. HNSW en feature flag optionnel. 3. Tester recall@100 sur dataset standard (SIFT1M). 4. Si recall < 95% → augmenter ef_search ou rester en brute-force. |
| **Detection** | Recall@100 sur SIFT1M benchmark |

### R6 — WAL : corruption en cas de power failure

| | |
|---|---|
| **Phase** | v0.1d / v0.5 |
| **Impact** | Critique |
| **Probabilite** | Basse |
| **Description** | Un power failure pendant un fsync peut corrompre le WAL. Le checksum ne suffit pas si le sector write est torn. |
| **Mitigation** | 1. CRC32 par frame WAL. 2. Checksums dans le file header. 3. Write-then-rename pour les operations critiques (compaction). 4. En v0.5, envisager double-write pour les pages critiques (technique SQLite). |
| **Detection** | Tests avec injection de crashes (kill -9 pendant fsync) |

### R7 — mmap + write = data corruption

| | |
|---|---|
| **Phase** | v0.5 |
| **Impact** | Critique |
| **Probabilite** | Basse |
| **Description** | mmap pour les ecritures est dangereux : un crash laisse des pages partiellement ecrites sans possibilite de rollback. |
| **Mitigation** | **Decision prise : JAMAIS de mmap pour les ecritures.** Les ecritures passent par le WAL (write() syscall). mmap est utilise uniquement en lecture. |
| **Detection** | Code review : interdire MAP_SHARED avec PROT_WRITE |

### R8 — Combinaison ACT-R + FSRS non validee

| | |
|---|---|
| **Phase** | v0.3 |
| **Impact** | Mineur |
| **Probabilite** | Haute |
| **Description** | La combinaison ACT-R (scoring temps-reel) + FSRS (scheduling) est inedite. Pas de litterature validant ce couplage. |
| **Mitigation** | 1. Les deux systemes sont orthogonaux (scoring vs scheduling) → pas de conflit theorique. 2. A/B testing avec des KG reels en v0.4. 3. Parametres configurables pour ajuster/desactiver chacun. |
| **Detection** | Tests empiriques sur des patterns d'utilisation reels |

---

## Risques ecosystem / integration

### R9 — napi-rs v3 breaking changes

| | |
|---|---|
| **Phase** | v0.1e |
| **Impact** | Majeur |
| **Probabilite** | Moyenne |
| **Description** | napi-rs v3 est relativement recent. Des breaking changes dans le macro API ou le build system pourraient casser le binding. |
| **Mitigation** | 1. Pin la version exacte dans Cargo.toml. 2. CI avec la version pinnee. 3. Tester sur Node 18, 20, 22. |
| **Detection** | CI multi-platform multi-node |

### R10 — ONNX Runtime linking issues

| | |
|---|---|
| **Phase** | v0.2+ (feature embedder) |
| **Impact** | Mineur |
| **Probabilite** | Moyenne |
| **Description** | ort (Rust ONNX) a des problemes recurrents de linking sur certaines plateformes, surtout Windows et musl Linux. |
| **Mitigation** | 1. Feature flag optionnel (pas dans default). 2. Strategie principale : embeddings externes (l'utilisateur fournit les vecteurs). 3. Documentation claire des pre-requis. |
| **Detection** | CI multiplateforme avec feature embedder |

### R11 — WASM : pas de mmap, pas de threads, pas de SIMD partiel

| | |
|---|---|
| **Phase** | v0.7 |
| **Impact** | Mineur |
| **Probabilite** | Haute |
| **Description** | En WASM, pas de mmap, pas de multi-thread, SIMD limite (pas de FMA). Le backend WASM sera significativement plus lent. |
| **Mitigation** | 1. Fallback read() au lieu de mmap. 2. Single-threaded OK (WASM est single-threaded de toute facon). 3. WASM SIMD 128-bit pour le cosine. 4. Budget : 3-5x plus lent qu'en natif = acceptable pour le browser. |
| **Detection** | Benchmarks WASM dedies |

---

## Risques projet

### R12 — Scope creep sur le modele bio-inspire

| | |
|---|---|
| **Impact** | Majeur |
| **Probabilite** | Haute |
| **Description** | Tentation d'ajouter toujours plus de mecanismes neuroscientifiques (amygdala, hippocampal replay detaille, etc.) qui complexifient sans apporter de valeur prouvee. |
| **Mitigation** | 1. Chaque mecanisme bio doit prouver sa valeur par un benchmark ou A/B test. 2. Si pas de gain mesurable → desactiver par defaut. 3. La phase v0.3 est le scope max pour le modele bio en v1.0. |

### R13 — Performance insuffisante vs SQLite

| | |
|---|---|
| **Impact** | Critique |
| **Probabilite** | Basse |
| **Description** | Si hora-graph-core est plus lent que SQLite pour du CRUD basique, le projet perd sa raison d'etre. |
| **Mitigation** | 1. Benchmarks comparatifs des la v0.1f. 2. Le CRUD in-memory est inherement plus rapide que SQLite (pas de SQL parsing). 3. Le storage embedded doit battre SQLite sur les graph operations. |
| **Detection** | Bench comparatif hora vs rusqlite a chaque release |

### R14 — Adoption : trop de concepts a comprendre

| | |
|---|---|
| **Impact** | Majeur |
| **Probabilite** | Moyenne |
| **Description** | Le modele bio-inspire (ACT-R, dark nodes, dream cycle) peut intimider les developpeurs. Ils veulent juste un KG rapide. |
| **Mitigation** | 1. Le mode par defaut est un KG classique (bio features desactivees). 2. Documentation progressive : "start simple, add bio later". 3. Le "3 lines to start" doit vraiment marcher sans comprendre ACT-R. |

---

## Matrice recapitulative

| ID | Risque | Impact | Proba | Phase | Mitigation principale |
|----|--------|--------|-------|-------|-----------------------|
| R1 | HashMap lent | Majeur | Moy | v0.1 | Benchmark + fallback BTreeMap |
| R2 | Serde binaire bugs | Critique | Haute | v0.1d | proptest + CRC32 |
| R3 | SIMD dispatch | Majeur | Basse | v0.2a | Tests croisees |
| R4 | Tokenizer simpliste | Mineur | Moy | v0.2b | Hybrid search compense |
| R5 | HNSW recall | Majeur | Moy | v0.2 | Brute-force fallback |
| R6 | WAL power failure | Critique | Basse | v0.5 | CRC32 + write-rename |
| R7 | mmap write | Critique | Basse | v0.5 | Interdit par design |
| R8 | ACT-R+FSRS | Mineur | Haute | v0.3 | A/B testing |
| R9 | napi-rs breaking | Majeur | Moy | v0.1e | Pin version |
| R10 | ONNX linking | Mineur | Moy | v0.2+ | Feature flag optionnel |
| R11 | WASM limitations | Mineur | Haute | v0.7 | Fallbacks documentes |
| R12 | Scope creep bio | Majeur | Haute | v0.3 | Benchmark-driven |
| R13 | Perf vs SQLite | Critique | Basse | v0.1 | Bench comparatif |
| R14 | Adoption | Majeur | Moy | v1.0 | Progressive disclosure |

---

*Document cree le 2026-03-02.*
