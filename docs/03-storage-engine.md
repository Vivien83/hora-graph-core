# 03 — Storage Engine

> Format .hora page-based, WAL, mmap, crash recovery.
> Inspire de SQLite, specialise pour knowledge graph.

---

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────┐
│                    Fichier .hora                              │
│                                                              │
│  ┌──────────┬──────────┬──────────┬──────────┬────────────┐ │
│  │  Header  │  Page    │  Pages   │  Pages   │   WAL      │ │
│  │  (100B)  │ Directory│  (data)  │  (free)  │  (.wal)    │ │
│  └──────────┴──────────┴──────────┴──────────┴────────────┘ │
│                                                              │
│  Chaque page = 4KB (configurable)                            │
│  Type de page determine par son header                       │
└─────────────────────────────────────────────────────────────┘
```

---

## File Header (page 0, premiere 100 bytes)

```rust
#[repr(C)]
pub struct FileHeader {
    // Magic + version (16 bytes)
    pub magic: [u8; 4],          // "HORA" (0x484F5241)
    pub format_version: u16,     // 1 pour v0.1 (incremente a chaque breaking change)
    pub min_read_version: u16,   // version minimale pour lire ce fichier
    pub reserved_1: [u8; 8],     // reserve pour compatibilite future

    // Geometry (16 bytes)
    pub page_size: u32,          // 4096 par defaut
    pub page_count: u32,         // nombre total de pages
    pub freelist_page: u32,      // premiere page de la freelist
    pub freelist_count: u32,     // nombre de pages libres

    // Schema (24 bytes)
    pub embedding_dims: u16,     // 384, 768, 1536...
    pub entity_type_count: u8,   // nombre de types d'entites
    pub relation_type_count: u8, // nombre de types de relations
    pub entity_count: u64,       // nombre total d'entites
    pub edge_count: u64,         // nombre total d'aretes
    pub episode_count: u32,      // nombre total d'episodes

    // Page directory pointers (24 bytes)
    pub entity_root_page: u32,   // racine du B+ tree des entites
    pub edge_root_page: u32,     // racine du B+ tree des aretes
    pub string_pool_page: u32,   // premiere page du string pool
    pub vector_root_page: u32,   // premiere page des vecteurs
    pub bm25_root_page: u32,     // racine de l'index inverse
    pub temporal_root_page: u32, // racine de l'index temporel

    // Integrity (16 bytes)
    pub last_checkpoint: i64,    // timestamp du dernier checkpoint WAL
    pub header_checksum: u32,    // CRC32 du header (sauf ce champ)
    pub reserved_2: [u8; 4],     // reserve

    // IDs (8 bytes)
    pub next_entity_id: u64,     // prochain ID libre
}
// Total: 104 bytes, dans la premiere page (4KB)
// Le reste de la page 0 contient les tables de types
```

### Tables de types (dans le reste de la page 0)

```
Offset 104: Entity types table
  count: u8
  entries: [
    { id: u8, name_len: u8, name: [u8; name_len] }
  ]

Offset 104 + entity_types_size: Relation types table
  count: u8
  entries: [
    { id: u8, name_len: u8, name: [u8; name_len] }
  ]
```

---

## Types de pages

```rust
#[repr(u8)]
pub enum PageType {
    Free = 0,           // Page libre (dans la freelist)
    EntityLeaf = 1,     // B+ tree leaf: entites triees par ID
    EntityInterior = 2, // B+ tree interior: pointeurs + separateurs
    EdgeData = 3,       // Bloc CSR d'aretes
    StringPool = 4,     // Strings contigues, append-only
    VectorData = 5,     // Vecteurs alignes 32B
    Bm25Posting = 6,    // Posting list pour un terme
    Bm25Dict = 7,       // Dictionnaire de termes
    TemporalIndex = 8,  // Index bi-temporel
    PropertyColumn = 9, // Colonne de proprietes
    EpisodeData = 10,   // Episodes
    ActivationLog = 11, // Log des activations
    Overflow = 12,      // Page overflow pour grandes donnees
}
```

### Page header commun (8 bytes)

```rust
#[repr(C)]
pub struct PageHeader {
    pub page_type: u8,      // PageType enum
    pub flags: u8,          // compression, dirty, etc.
    pub item_count: u16,    // nombre d'items dans cette page
    pub checksum: u32,      // CRC32 du contenu (hors header)
}
// Donnees utiles par page: 4096 - 8 = 4088 bytes
```

---

## B+ Tree (pour Entity et Edge index)

### Pourquoi B+ tree ?

| Structure | Lecture seq. | Lookup ID | Insert | Cache perf |
|-----------|-------------|-----------|--------|------------|
| HashMap | O(n) | O(1) amorti | O(1) | Mauvais |
| B-tree | O(n) | O(log n) | O(log n) | Bon |
| **B+ tree** | **O(n) sequentiel** | **O(log n)** | **O(log n)** | **Excellent** |

B+ tree : les valeurs sont dans les feuilles, les noeuds internes ne contiennent que les cles. Les feuilles sont chainees pour le scan sequentiel.

### Entity B+ tree layout

```
Interior page:
  ┌────────────────────────────────────────────┐
  │ PageHeader (8B)                            │
  │ child_0 (4B) | key_0 (8B) | child_1 (4B)  │
  │ key_1 (8B) | child_2 (4B) | ...            │
  └────────────────────────────────────────────┘
  Capacite: (4088 - 4) / (8 + 4) ≈ 340 cles par page interior

Leaf page:
  ┌────────────────────────────────────────────┐
  │ PageHeader (8B)                            │
  │ prev_leaf (4B) | next_leaf (4B)            │
  │ Entity_0 (48B) | Entity_1 (48B) | ...      │
  └────────────────────────────────────────────┘
  Capacite: (4088 - 8) / 48 ≈ 85 entites par page leaf

Avec 340 cles/interior et 85 entites/leaf :
  - 1 niveau : 85 entites
  - 2 niveaux : 340 × 85 = 28,900 entites
  - 3 niveaux : 340 × 340 × 85 = 9.8M entites
  - 4 niveaux : 340^3 × 85 = 3.3B entites

→ 3 niveaux suffisent pour < 10M entites (99% des use cases)
→ 3 lectures de page max pour trouver n'importe quelle entite
```

---

## WAL (Write-Ahead Log)

### Principe

Toute ecriture va d'abord dans le WAL. Le fichier principal n'est modifie qu'au checkpoint.

```
Flux d'ecriture :
  1. Serialiser la modification
  2. Ecrire le frame WAL
  3. fsync le WAL
  4. Retourner succes a l'appelant

Flux de lecture :
  1. Chercher dans le WAL (modifications recentes)
  2. Si pas trouve → lire dans le fichier principal

Checkpoint (periodique ou explicit) :
  1. Pour chaque frame WAL, ecrire la page dans le fichier principal
  2. fsync le fichier principal
  3. Truncate le WAL
```

### WAL frame format

```rust
#[repr(C)]
pub struct WalHeader {
    pub magic: [u8; 4],         // "WLOG"
    pub version: u16,           // 1
    pub page_size: u32,         // doit matcher le fichier principal
    pub checkpoint_seq: u64,    // numero de sequence du dernier checkpoint
    pub salt: [u8; 8],          // random, change a chaque checkpoint
}

#[repr(C)]
pub struct WalFrame {
    pub page_number: u32,       // quelle page est modifiee
    pub db_size: u32,           // taille de la DB au moment de cette ecriture
    pub salt: [u8; 8],          // doit matcher le header
    pub frame_checksum: u32,    // CRC32 du frame + donnees
    // suivi de page_size bytes de donnees
}
```

### WAL et concurrency

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Reader 1 │     │ Reader 2 │     │  Writer  │
│ (snap=5) │     │ (snap=7) │     │  (WAL)   │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │
     │   fichier .hora (checkpoint 5)  │
     │──────────────────────────────── │
     │                │                │
     │                │   WAL frames 6,7,8,9
     │                │────────────────│
     │                                 │

Reader 1 : lit le fichier principal (snapshot 5)
Reader 2 : lit fichier + WAL frames 6-7 (snapshot 7)
Writer  : ecrit dans le WAL (frame 8, 9...)
→ Pas de conflit. Readers isolees du writer.
```

### Taille max du WAL

**Politique :** checkpoint automatique quand WAL > 1000 frames (~4MB pour pages 4KB).
Configurable via `HoraConfig::wal_auto_checkpoint`.

---

## mmap Strategy

### Quand utiliser mmap

| Situation | Strategie | Raison |
|-----------|-----------|--------|
| Lecture de pages existantes | **mmap** | Zero-copy, OS gere le cache |
| Ecriture | **write()** via WAL | mmap + ecriture = dangereux (crash = corruption) |
| WASM | **read()** fallback | mmap n'existe pas en WASM |
| Petits fichiers (< 1MB) | **read()** | Overhead mmap pas justifie |

### Implementation

```rust
pub struct MmapReader {
    mmap: memmap2::Mmap,  // Note: memmap2 est une dep build-time acceptable
    page_size: usize,
}

impl MmapReader {
    pub fn read_page(&self, page_num: u32) -> &[u8] {
        let offset = page_num as usize * self.page_size;
        &self.mmap[offset..offset + self.page_size]
    }
}
```

### Alternative sans memmap2

Si on veut vraiment zero dep, on peut utiliser directement :
```rust
// Unix
let ptr = libc::mmap(null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0);

// Mais on perd la portabilite Windows...
```

**Decision :** Accepter `memmap2` comme build-time dep (c'est un wrapper mince autour de l'OS). Fournir un fallback `read()` pour WASM et `--no-default-features`.

---

## Crash Recovery

### Scenarios de crash

| Scenario | Etat | Recovery |
|----------|------|----------|
| Crash pendant ecriture WAL | Frame incomplet | WAL frame checksum invalide → ignorer |
| Crash pendant checkpoint | Pages partiellement ecrites | Rejouer le WAL depuis le dernier checkpoint valide |
| Crash pendant compact | Fichier partiellement reecrit | Compact ecrit dans un nouveau fichier, rename atomique |

### Sequence de recovery a l'ouverture

```
fn open(path):
  1. Lire le file header
  2. Verifier magic "HORA" + version
  3. Si WAL existe :
     a. Lire WAL header, verifier magic "WLOG"
     b. Scanner les frames, verifier checksums
     c. Construire un index : page_num → dernier frame valide
     d. C'est le "WAL index" (en memoire)
  4. Verifier header_checksum
  5. La DB est prete (lectures combinent fichier + WAL index)
```

---

## Freelist

### Design

La freelist est une liste chainee de pages libres :

```
Freelist page:
  ┌────────────────────────────────────────────┐
  │ PageHeader (8B, type=Free)                 │
  │ next_freelist_page: u32 (4B)               │
  │ count: u16 (2B)                            │
  │ free_page_ids: [u32; count]                │
  └────────────────────────────────────────────┘
  Capacite: (4088 - 6) / 4 ≈ 1020 page IDs par freelist page
```

Quand on delete une entite → ses pages vont dans la freelist.
Quand on insere → prendre depuis la freelist avant d'allouer en fin de fichier.

---

## Compaction (VACUUM)

### Probleme

Apres beaucoup de deletes, le fichier a des trous (pages dans la freelist). Le fichier ne retrecit pas automatiquement.

### Strategie : incremental vacuum

```
1. Identifier la derniere page utilisee (non-free)
2. Identifier la premiere page libre
3. Deplacer le contenu de la derniere page vers la premiere page libre
4. Mettre a jour les pointeurs (B+ tree, CSR offsets, string pool refs)
5. Reduire page_count de 1
6. Truncate le fichier
7. Repeter
```

**Avantage :** Pas besoin de recopier tout le fichier.
**Inconvenient :** Plus lent qu'un full vacuum pour beaucoup de fragmentation.

### Full vacuum (rebuild)

```
1. Creer un nouveau fichier .hora.tmp
2. Scanner toutes les entites/edges/strings/vecteurs
3. Reecrire dans le nouveau fichier (compact, defragmente)
4. rename(.hora.tmp, .hora) — atomique sur Unix
```

**Quand :** utilise manuellement via `hora.compact()`, ou si freelist > 30% du fichier.

---

## Transactions (simplifiees)

### Pas de transactions multi-statement completes en v0.1

En v0.1, chaque operation est atomique individuellement (grace au WAL). Les transactions multi-statement arrivent en v0.5.

### Atomicite par operation

```
hora.add_entity() :
  1. Debut transaction implicite
  2. Ecrire entity page → WAL frame
  3. Ecrire string pool → WAL frame
  4. Ecrire BM25 index → WAL frame
  5. Commit (fsync WAL)
  → Si crash entre 2 et 5 : au recovery, frame checksums invalides
  → Les frames partiels sont ignores → comme si l'operation n'avait pas eu lieu
```

---

## StorageOps trait

```rust
pub trait StorageOps {
    // CRUD Entities
    fn put_entities(&mut self, batch: &[Entity]) -> Result<Vec<EntityId>>;
    fn get_entity(&self, id: EntityId) -> Result<Option<Entity>>;
    fn update_entity(&mut self, id: EntityId, update: EntityUpdate) -> Result<()>;
    fn delete_entity(&mut self, id: EntityId) -> Result<()>;
    fn scan_entities(&self) -> Result<EntityIterator>;

    // CRUD Edges
    fn put_edges(&mut self, batch: &[Edge]) -> Result<Vec<EdgeId>>;
    fn get_edge(&self, id: EdgeId) -> Result<Option<Edge>>;
    fn scan_edges(&self, source: EntityId) -> Result<EdgeIterator>;
    fn delete_edge(&mut self, id: EdgeId) -> Result<()>;

    // Search
    fn vector_search(&self, query: &[f32], k: usize, filter: Option<&Filter>) -> Result<Vec<SearchHit>>;
    fn text_search(&self, query: &str, k: usize) -> Result<Vec<SearchHit>>;

    // Temporal
    fn facts_at(&self, timestamp: i64) -> Result<Vec<Edge>>;
    fn entity_history(&self, id: EntityId) -> Result<Vec<Edge>>;

    // Lifecycle
    fn flush(&mut self) -> Result<()>;
    fn checkpoint(&mut self) -> Result<()>;
    fn snapshot(&self, path: &Path) -> Result<()>;
    fn compact(&mut self) -> Result<()>;

    // Stats
    fn stats(&self) -> Result<StorageStats>;
}
```

### Implementations

| Backend | Status | Dependencies | Use case |
|---------|--------|-------------|----------|
| `MemoryStorage` | v0.1 | aucune | Tests, ephemere |
| `EmbeddedStorage` | v0.1 basique, v0.5 complet | memmap2 (optionnel) | Production par defaut |
| `SqliteStorage` | v0.6 | rusqlite | Ecosysteme SQLite existant |
| `PostgresStorage` | v0.6 | tokio-postgres | Multi-client, scale |

---

## Benchmarks cibles storage

| Operation | Memory | Embedded v0.1 | Embedded v0.5 | SQLite | Postgres |
|-----------|--------|--------------|---------------|--------|----------|
| Insert 1 entity | < 100ns | < 1μs | < 1μs | < 10μs | < 100μs |
| Get entity by ID | < 50ns | < 500ns | < 200ns (mmap) | < 5μs | < 500μs |
| Scan 1K entities | < 10μs | < 50μs | < 20μs (mmap) | < 100μs | < 1ms |
| BFS 3-hop | < 100μs | < 500μs | < 200μs | < 1ms | < 5ms |
| Open 1M entities | instant | < 500ms | < 50ms (mmap) | < 100ms | N/A |
| Checkpoint | N/A | < 10ms | < 10ms | N/A | N/A |

---

*Document cree le 2026-03-02. Fait partie de la preparation hora-graph-core.*
