# 02 — Data Layout

> Structures binaires, formats memoire, alignement.
> Chaque octet compte. Chaque decision est justifiee.

---

## Philosophie

1. **Compact** — minimiser l'empreinte memoire et disque
2. **Aligne** — respecter les alignements CPU pour eviter les penalites
3. **Cache-friendly** — structures contigues, acces sequentiel
4. **Extensible** — pouvoir ajouter des champs sans casser le format

---

## Entity

### Layout memoire (48 bytes, aligne 8B)

```rust
#[repr(C)]
pub struct Entity {
    // --- Identite (11 bytes logiques, 16 avec padding) ---
    pub id: u64,              // 8B  — auto-increment, jamais reutilise
    pub entity_type: u8,      // 1B  — index dans la table des types (max 256)
    _pad1: [u8; 3],           // 3B  — padding pour alignement u32
    pub name_offset: u32,     // 4B  — pointeur dans le string pool

    // --- Donnees (10 bytes) ---
    pub name_len: u16,        // 2B  — longueur du nom (max 65535 chars)
    pub properties_offset: u32, // 4B — pointeur vers la page de proprietes (0 = aucune)
    pub embedding_offset: u32, // 4B — pointeur vers la page de vecteurs (0 = aucun)

    // --- Temporel (16 bytes) ---
    pub created_at: i64,      // 8B  — unix millis, quand l'entite a ete creee
    pub last_seen: i64,       // 8B  — unix millis, dernier acces/modification

    // --- Graph (6 bytes logiques, 8 avec padding) ---
    pub adjacency_offset: u32, // 4B — pointeur CSR vers la liste d'aretes
    pub adjacency_count: u16, // 2B  — nombre d'aretes sortantes (max 65535)
    _pad2: [u8; 2],           // 2B  — padding pour alignement f32

    // --- Bio-inspired (4 bytes) ---
    pub activation: f32,      // 4B  — score ACT-R courant
}
// Total: 48 bytes (avec padding pour alignement naturel)
// 1M entites = 48 MB
// 10M entites = 480 MB
```

### Justifications

| Champ | Taille | Pourquoi |
|-------|--------|----------|
| `id: u64` | 8B | Support > 4 milliards d'entites. u32 limiterait a ~4M. |
| `entity_type: u8` | 1B | 256 types suffisent. Si besoin de plus → refactor en u16. |
| `name_offset: u32` | 4B | Pointe dans le string pool. 4GB de strings max. |
| `name_len: u16` | 2B | Noms > 64KB n'ont pas de sens pour un KG. |
| `properties_offset: u32` | 4B | 0 = pas de proprietes. Proprietes stockees separement (columnar). |
| `embedding_offset: u32` | 4B | 0 = pas d'embedding. Vecteurs stockes dans des pages alignees. |
| `created_at: i64` | 8B | Milliseconde precision. Couvre ±292M annees. |
| `last_seen: i64` | 8B | Mis a jour a chaque acces. Sert au decay ACT-R. |
| `adjacency_offset: u32` | 4B | Pointeur CSR. Aretes stockees contiguement. |
| `adjacency_count: u16` | 2B | 65K aretes sortantes max. Suffisant pour un KG. |
| `activation: f32` | 4B | Score ACT-R. f32 = precision suffisante, aligne naturellement. |

### Alternatives rejetees

| Alternative | Pourquoi rejetee |
|------------|------------------|
| `id: u32` | 4M entites max — insuffisant pour un KG d'entreprise |
| `activation: f64` | Precision inutile, double la taille |
| Proprietes inline | Taille variable → casse l'alignement et le packing |
| `HashMap<String, Value>` props | Overhead enorme, pas serialisable efficacement |

---

## Edge (Fact)

### Layout memoire (40 bytes, aligne 8B)

```rust
#[repr(C)]
pub struct Edge {
    // --- Identite (8 bytes) ---
    pub id: u64,              // 8B  — auto-increment unique

    // --- Relation (10 bytes logiques, 16 avec padding) ---
    pub source: u64,          // 8B  — EntityId source
    pub target: u64,          // 8B  — EntityId cible
    pub relation: u8,         // 1B  — index dans la table des types de relation
    pub confidence: u8,       // 1B  — 0-255 mappe sur 0.0-1.0 (precision ~0.004)

    // --- Bi-temporal (24 bytes) ---
    pub valid_at: i64,        // 8B  — quand ce fait est devenu vrai (monde reel)
    pub invalid_at: i64,      // 8B  — quand ce fait a cesse d'etre vrai (0 = toujours valide)
    pub created_at: i64,      // 8B  — quand on a enregistre ce fait (systeme)

    // --- Donnees (8 bytes) ---
    pub description_offset: u32, // 4B — string pool (description textuelle du fait)
    pub metadata_offset: u32, // 4B  — page de metadata (0 = aucune)
}
// Total: 58 bytes logiques → 64 bytes avec alignement
// Ou on peut compacter a 56 bytes en reordonnant

// Alternative compacte si on retire l'id explicite :
// source(8) + target(8) + relation(1) + confidence(1) + pad(6) +
// valid_at(8) + invalid_at(8) + created_at(8) + desc(4) + meta(4) = 56B
```

### Decision : Edge ID explicite ou implicite ?

| Approche | Avantage | Inconvenient |
|----------|----------|-------------|
| **ID explicite (u64)** | Reference directe, invalidation par ID | +8B par edge |
| **ID implicite (offset)** | Economie 8B | Invalidation complexe, fragmentation |

**Decision : ID explicite.** La bi-temporalite necessite des references stables pour `invalidate_fact(id)`.

### CSR (Compressed Sparse Row) pour les adjacences

```
Pour l'entite E avec adjacency_offset=X et adjacency_count=N :

Memory[X .. X + N*8] contient N x u64 (edge IDs)
   ┌──────┬──────┬──────┬──────┐
   │ eid1 │ eid2 │ eid3 │ ...  │  ← edge IDs, tries par target
   └──────┴──────┴──────┴──────┘

Les Edge structs sont stockees dans un tableau separe, indexe par edge ID.
CSR donne O(1) pour "toutes les aretes de E" + cache-friendly sequential scan.
```

### Pourquoi CSR et pas Adjacency List classique

| Structure | Acces toutes aretes | Ajout arete | Cache perf | Memoire |
|-----------|-------------------|-------------|------------|---------|
| CSR | O(1) + scan sequentiel | O(n) realloc | Excellent | Minimal |
| Adj. List (Vec) | O(1) + scan | O(1) amortized | Bon | +24B/entity overhead |
| HashMap | O(1) + iter | O(1) amortized | Mauvais | Enorme overhead |

**Decision : CSR pour la lecture (90% des ops), avec un write buffer pour les ajouts.**

Le write buffer est un `Vec<(EntityId, EdgeId)>` trie periodiquement et fusionne dans le CSR lors du flush/compact.

---

## Episode

### Layout memoire

```rust
#[repr(C)]
pub struct Episode {
    pub id: u64,              // 8B
    pub source_type: u8,      // 1B  — conversation, document, api, manual...
    pub timestamp: i64,       // 8B  — quand cet episode a eu lieu
    pub content_offset: u32,  // 4B  — string pool (texte brut de l'episode)
    pub content_len: u32,     // 4B  — longueur du contenu
    pub entity_refs_offset: u32, // 4B — liste d'EntityId references
    pub entity_refs_count: u16, // 2B — nombre d'entites referencees
    pub fact_refs_offset: u32, // 4B — liste d'EdgeId crees par cet episode
    pub fact_refs_count: u16, // 2B — nombre de faits crees
    pub consolidation_count: u8, // 1B — combien de fois consolide (pour CLS)
}
// ~38 bytes
```

**Role :** Store fast-write pour les evenements bruts. Analogie cerveau : hippocampe (memoire episodique). Les episodes sont progressivement transformes en faits semantiques via le dream cycle (CLS transfer).

---

## String Pool

### Design

```
┌──────────────────────────────────────────────┐
│  Offset 0:  "hora-engine\0"                  │  len=12
│  Offset 12: "depends_on\0"                   │  len=11
│  Offset 23: "Vivien MARTIN\0"                │  len=14
│  ...                                         │
└──────────────────────────────────────────────┘
```

- Strings null-terminees pour compatibilite C FFI
- Append-only (on n'efface jamais un string, compaction le fait)
- Dedup optionnelle : hash table `hash(string) → offset` pour eviter les doublons
- Encodage : UTF-8 strict

### Pourquoi un pool et pas inline ?

| Approche | Taille Entity | Flexibilite | Cache |
|----------|---------------|-------------|-------|
| Inline (fixed 64B) | +64B/entity | Tronque | Entity pages polluees |
| Inline (variable) | Variable | Illimitee | Impossible a indexer efficacement |
| **Pool (offset+len)** | +6B/entity | Illimitee | Entities compactes, strings separees |

---

## Property Storage (columnar)

### Design

Les proprietes sont stockees en format **columnar** plutot que row-based :

```
Schema: { "language": String, "version": String, "stars": Int }

Column "language":
  ┌───────┬──────┬───────────┬──────────────┐
  │ eid=1 │ "ts" │ eid=2     │ "rust"       │  ...
  └───────┴──────┴───────────┴──────────────┘

Column "version":
  ┌───────┬───────┬───────────┬──────────────┐
  │ eid=1 │ "5.0" │ eid=2     │ "1.80"       │  ...
  └───────┴───────┴───────────┴──────────────┘

Column "stars":
  ┌───────┬──────┬───────────┬──────────────┐
  │ eid=1 │ 42   │ eid=2     │ 100          │  ...
  └───────┴──────┴───────────┴──────────────┘
```

### Pourquoi columnar ?

1. **Compression** — valeurs du meme type compressent mieux ensemble
2. **Scan** — "toutes les entites avec stars > 50" ne lit qu'une colonne
3. **Sparse** — pas de gaspillage pour les proprietes absentes
4. **Extensible** — ajouter une nouvelle propriete = ajouter une colonne

### Types de proprietes supportes

```rust
pub enum PropertyValue {
    Null,           // 1B tag
    Bool(bool),     // 1B tag + 1B
    Int(i64),       // 1B tag + 8B
    Float(f64),     // 1B tag + 8B
    String(u32),    // 1B tag + 4B (offset dans string pool)
    Bytes(u32, u32), // 1B tag + 4B offset + 4B len
}
```

---

## Vector Storage

### Alignement SIMD

```
AVX2 requiert alignement 32 bytes.
Un vecteur 384-dim f32 = 384 × 4 = 1536 bytes.
1536 / 32 = 48 → naturellement aligne si le debut est aligne 32B.

Page layout pour vecteurs (4096B page):
  ┌──────────────────────────┐
  │ Page header (32B)        │  ← aligne 32B
  │ Vector 0 (1536B)         │  ← aligne 32B (offset 32)
  │ Vector 1 (1536B)         │  ← aligne 32B (offset 1568)
  │ Padding (992B)           │  ← espace perdu
  └──────────────────────────┘
  Utilisation: 2 vecteurs / page 4KB = 75%

Alternative avec page 8KB :
  ┌──────────────────────────┐
  │ Page header (32B)        │
  │ Vector 0 (1536B)         │
  │ Vector 1 (1536B)         │
  │ Vector 2 (1536B)         │
  │ Vector 3 (1536B)         │
  │ Vector 4 (1536B)         │
  │ Padding (288B)           │
  └──────────────────────────┘
  Utilisation: 5 vecteurs / page 8KB = 96%
```

**Decision a prendre :** page 4KB (standard, 75% utilisation) ou 8KB (meilleur packing vecteurs) ?

**Recommandation :** Page 4KB par defaut, configurable. Le packing sous-optimal est compense par le fait que 4KB est la taille de page OS standard (meilleur mmap).

### Layout pour differentes dimensions

| Dimensions | Bytes/vec | Vecs/page 4KB | Utilisation |
|-----------|-----------|---------------|-------------|
| 384 | 1536B | 2 | 75% |
| 768 | 3072B | 1 | 75% |
| 1536 | 6144B | 0 (overflow!) | Necessite page > 4KB |

**Note :** Pour dims=1536 (OpenAI), il faut soit des pages 8KB soit des overflow pages. A gerer dans la config.

---

## Activation Log

### Pour le calcul ACT-R

Le calcul B(i) = ln(Σ tⱼ^(-d)) necessite l'historique des acces.

**Approche naive :** stocker tous les timestamps → memoire O(n_acces) par entite.
**Approche optimisee :** approximation running avec counter + dernier acces.

```rust
pub struct ActivationRecord {
    pub entity_id: u64,       // 8B
    pub access_count: u32,    // 4B  — nombre total d'acces
    pub last_access: i64,     // 8B  — dernier acces (millis)
    pub creation_time: i64,   // 8B  — premier acces (millis)
    pub cached_activation: f32, // 4B — dernier score calcule
    pub is_dirty: bool,       // 1B  — besoin de recalcul ?
}
```

**Approximation Anderson & Schooler (1991) :**

Au lieu de stocker chaque timestamp, on peut approximer B(i) avec :

```
B(i) ≈ ln(n) + (1-d) × ln(L/n)
```

Ou `n` = nombre d'acces, `L` = age total, `d` = decay (0.5).
Precision ~95% par rapport au calcul exact. Enorme gain memoire.

**Decision :** Utiliser l'approximation par defaut, avec option de log complet pour la recherche/debug.

---

## Resume des tailles

| Structure | Taille unitaire | 100K | 1M | 10M |
|-----------|----------------|------|-----|------|
| Entity | 48B | 4.8 MB | 48 MB | 480 MB |
| Edge | 56-64B | 6.4 MB | 64 MB | 640 MB |
| Episode | 38B | 3.8 MB | 38 MB | 380 MB |
| Activation | 33B | 3.3 MB | 33 MB | 330 MB |
| Vector (384d) | 1536B | 153 MB | 1.5 GB | 15 GB |
| BM25 index | ~10B/token | variable | variable | variable |
| String pool | ~30B/nom moyen | 3 MB | 30 MB | 300 MB |

**Observation :** Les vecteurs dominent largement. Pour 1M entites avec embeddings, il faut ~1.5GB rien que pour les vecteurs. C'est un argument fort pour :
1. Ne pas forcer l'embedding (optionnel)
2. Considerer la quantization (f32 → f16 ou int8) en v0.5+

---

*Document cree le 2026-03-02. Fait partie de la preparation hora-graph-core.*
