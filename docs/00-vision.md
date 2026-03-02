# 00 — Vision & Positionnement

> Le SQLite du knowledge graph pour l'ere des agents IA.
> Moteur bio-inspire, embedded, zero-dependency, Rust.

---

## Probleme

Les agents IA ont besoin de memoire persistante structuree. Les solutions existantes :
- **Requierent une infra externe** (Neo4j, PostgreSQL, Redis, Qdrant)
- **Sont verrouillees a un langage** (Python-only pour Zep, Cognee, LangMem)
- **N'ont pas de modele temporel** (Mem0, LangMem — pas de bi-temporalite)
- **Ne modelisent pas l'oubli** (tout est permanent, rien ne decay)
- **Sont trop complexes a deployer** (Microsoft GraphRAG: $33K d'indexation)

Aucune solution ne combine : **embedded + temporal + hybrid search + bio-inspired + Rust + TS-native**.

---

## Solution

`hora-graph-core` : un fichier, zero config, zero dependance.

```
┌─────────────────────────────────────────┐
│         Votre application / agent       │
├─────────────────────────────────────────┤
│    hora-graph-core (native binding)     │
├─────────────────────────────────────────┤
│         Un seul fichier .hora           │
└─────────────────────────────────────────┘
```

Comme SQLite est aux bases de donnees relationnelles,
hora-graph-core est aux knowledge graphs.

---

## Principes fondateurs

| # | Principe | Signification |
|---|---------|---------------|
| 1 | **ZERO DEPENDENCY** | stdlib Rust + SIMD intrinsics uniquement. Aucun crate runtime. |
| 2 | **PORTABLE** | Compile et tourne partout : macOS ARM/x64, Linux, Windows, WASM |
| 3 | **FAST** | SIMD vectorise, CSR cache-friendly, O(1) par hop, sub-milliseconde |
| 4 | **UNIVERSAL** | Bindings first-class : Node.js (napi-rs), Python (PyO3), WASM, C FFI |
| 5 | **CONFIGURABLE** | Storage, dimensions, types, relations, activation — tout pluggable |
| 6 | **BIO-INSPIRED** | Copie le cerveau humain, pas Neo4j. Oubli actif, consolidation, decay. |

---

## Positionnement concurrentiel

### Case vide identifiee

```
                    Embedded    Temporal    Hybrid     Bio-       Multi-
                    (1 file)    (bi-temp)   Search     inspired   Language
────────────────────────────────────────────────────────────────────────
Mem0                  ✗           ✗          ~          ✗          ~
Zep/Graphiti          ✗           ✓          ✓          ✗          ✗
Cognee                ✓           ✗          ✓          ✗          ✗
CozoDB                ✓           ✗          ~          ✗          ~
Kuzu                  ✓           ✗          ✗          ✗          ~
LangMem               ✗           ✗          ✗          ✗          ✗
MS GraphRAG           ✗           ✗          ✓          ✗          ✗
────────────────────────────────────────────────────────────────────────
hora-graph-core       ✓           ✓          ✓          ✓          ✓
```

### Concurrents detailles

| Solution | Forces | Faiblesses critiques |
|----------|--------|---------------------|
| **Mem0** | Plug-and-play, 24+ backends | 3 infras externes, pas de temporal, graph = paywall |
| **Zep/Graphiti** | Bi-temporal rigoureux, hybrid search | Neo4j obligatoire, Python-only |
| **Cognee** | Zero-config (Kuzu+LanceDB+SQLite) | Pas de temporal, Python-only |
| **CozoDB** | Rust, embedded, HNSW | Dev ralenti, Datalog = friction, pas de temporal |
| **Kuzu** | Embedded, Cypher, performant | OLAP-oriented, pas de temporal, pas de vector search |
| **LangMem** | Integration LangChain | Pas de graphe, purement vectoriel |
| **MS GraphRAG** | Community summaries | $33K indexing, pas incremental, pas embedded |

### Notre avantage defensible

1. **Bio-inspired** — Aucun concurrent ne modelise ACT-R, dark nodes, dream cycle
2. **Single-file** — Deployment frictionless, pas d'infra
3. **Rust + multi-binding** — Performance native dans n'importe quel langage
4. **Bi-temporal natif** — Pas un ajout, c'est dans le core

---

## Audience cible

| Utilisateur | Besoin | hora repond |
|------------|--------|-------------|
| Developpeur d'agents IA | Memoire persistante sans infra | Fichier .hora, npm install |
| Startup IA | KG scalable, pas cher | Embedded, zero infra cost |
| Chercheur en cognition | Modele de memoire computationnel | ACT-R, CLS, formules publiees |
| Projet open-source | Memoire pour chatbot/assistant | MIT, zero dep, facile a integrer |
| Entreprise | KG auditable avec temporalite | Bi-temporal, ACID, snapshot |

---

## Metriques de succes v1.0

| Metrique | Cible |
|----------|-------|
| Performance | 10x plus rapide que Zep/Graphiti sur les operations courantes |
| Taille | < 1MB binaire compile (sans embeddings) |
| Simplicite | 3 lignes de code pour ouvrir + ajouter + chercher |
| Portabilite | Fonctionne sur macOS, Linux, Windows, WASM |
| Fiabilite | 0 UB (miri clean), 0 memory leak (valgrind clean) |
| Adoption | npm install + pip install fonctionnels |

---

## Ce que hora-graph-core n'est PAS

- **Pas un remplacement de Neo4j** — pas de Cypher, pas de OLAP, pas de cluster
- **Pas un vector database** — les vecteurs servent la recherche hybride, pas le RAG pur
- **Pas un LLM framework** — pas d'integration LLM, pas de prompt engineering
- **Pas un service cloud** — c'est une librairie embedded, pas une API
- **Pas opinionate sur l'embedding** — accepte n'importe quels vecteurs, n'impose aucun modele

---

*Document cree le 2026-03-02. Fait partie de la preparation hora-graph-core.*
