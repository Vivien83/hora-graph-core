# Research: Format de fichier page-based

> Resultats de la recherche sur le design de formats binaires page-based (SQLite reference).

---

## References etudiees

- SQLite file format : https://www.sqlite.org/fileformat2.html
- SQLite WAL : https://www.sqlite.org/wal.html
- LMDB design : https://www.symas.com/lmdb
- WiredTiger (MongoDB) storage engine
- LevelDB/RocksDB LSM tree

---

## SQLite comme modele

### Pourquoi SQLite est la reference

- Format stable depuis 2004 (backward compatible)
- Single file, zero config
- WAL pour crash recovery
- mmap pour reads
- Utilise partout (3 trillion+ bases actives)

### Ce qu'on emprunte

| Feature | SQLite | hora |
|---------|--------|------|
| Page-based | 4KB pages | 4KB pages (configurable) |
| B+ tree | Pour tables et index | Pour entities et index temporel |
| WAL | Write-ahead logging | Identique |
| Freelist | Pages libres chainées | Identique |
| Magic header | "SQLite format 3\000" | "HORA" (4 bytes) |
| Checkpoint | WAL → main file | Identique |
| Locking | 5 lock states | Simplifie : RwLock |

### Ce qu'on fait differemment

| Feature | SQLite | hora |
|---------|--------|------|
| Schema | SQL DDL dynamique | Schema fixe (entity/edge/episode) |
| Query | SQL parser + VM | API directe (pas de SQL) |
| Types | Affinity-based | Types fixes structs |
| Vector data | N/A | Pages VectorData alignees 32B |
| BM25 | FTS5 extension | Index inverse natif |
| CSR edges | N/A | Pages EdgeData compactes |

---

## Page design

### Taille de page : 4KB

| Taille | Avantages | Inconvenients |
|--------|-----------|---------------|
| 512B | Granularite fine | Trop de pages, overhead headers |
| 2KB | Bon compromis | Peu standard |
| **4KB** | **Aligne avec page OS, SSD sector** | **Standard** |
| 8KB | Moins de pages | Plus de gaspillage |
| 16KB | Bon pour gros records | Trop gros pour petites donnees |

4KB est le choix par defaut (comme SQLite). Configurable pour les use cases specifiques (ex: 8KB pour les pages de vecteurs 384d = 2 vecteurs par page sans gaspillage).

### Page header : 8 bytes

```
[type: u8][flags: u8][item_count: u16][checksum: u32]
```

Overhead = 8/4096 = 0.2%. Negligeable.

### Capacites par type de page

| Page type | Item size | Items/page | Notes |
|-----------|-----------|-----------|-------|
| EntityLeaf | 48B | 85 | B+ tree leaf |
| EntityInterior | 12B (key+ptr) | 340 | B+ tree interior |
| EdgeData | 56B | 72 | CSR block |
| StringPool | variable | ~100-400 | Append-only strings |
| VectorData (384d) | 1536B | 2 | Aligne 32B pour AVX2 |
| Bm25Posting | variable (VByte) | ~500-2000 | Posting list |
| TemporalIndex | 16B | 255 | (valid_at, edge_id) |
| ActivationLog | ~50B | 81 | Activation states |

---

## WAL Design

### Principes
1. Toute ecriture va dans le WAL d'abord
2. Le fichier principal n'est modifie qu'au checkpoint
3. Crash = WAL frames invalides ignores = comme si l'ecriture n'avait pas eu lieu

### WAL frame format
```
[page_number: u32][db_size: u32][salt: [u8;8]][checksum: u32][page_data: page_size bytes]
```

### WAL index (in-memory)
HashMap<page_number, offset_in_wal> construit a l'ouverture.
Readers consultent le WAL index pour trouver les pages modifiees.

### Checkpoint strategies

| Mode | Description | Quand |
|------|------------|-------|
| Passive | Checkpoint quand aucun reader actif | Automatique (WAL > 1000 frames) |
| Full | Attendre que les readers finissent, puis checkpoint | Avant backup |
| Restart | Full + truncate WAL | Nettoyage periodique |
| Truncate | Restart + reset WAL header | Apres compact |

### WAL size limits
- Auto-checkpoint a 1000 frames (~4MB pour 4KB pages)
- Configurable via `HoraConfig::wal_auto_checkpoint`
- Si le WAL grandit trop (readers qui bloquent) → warning log

---

## mmap Strategy

### Read-only mmap
```rust
// Safe : le fichier principal n'est modifie que par checkpoint (rare)
let mmap = unsafe { memmap2::Mmap::map(&file) }; // PROT_READ, MAP_PRIVATE
```

### Pas de mmap pour les ecritures
**Raison :** un crash pendant un write mmap laisse des pages partiellement ecrites sans possibilite de rollback. Le WAL resout ce probleme.

### Re-mmap
Apres un checkpoint (le fichier a grandi) → unmap + re-mmap.
Ou utiliser `MmapOptions::offset()` pour mapper les nouvelles pages incrementalement.

### WASM fallback
```rust
#[cfg(target_arch = "wasm32")]
fn read_page(&self, page_num: u32) -> Vec<u8> {
    // pread() au lieu de mmap
    let mut buf = vec![0u8; self.page_size];
    self.file.seek(SeekFrom::Start(offset))?;
    self.file.read_exact(&mut buf)?;
    buf
}
```

---

## ACID Properties

| Property | Comment on l'assure |
|----------|-------------------|
| **Atomicity** | WAL : tout ou rien. Frame incomplet = ignore. |
| **Consistency** | CRC32 sur chaque page. Magic + version dans le header. |
| **Isolation** | WAL index : readers voient un snapshot coherent. |
| **Durability** | fsync sur le WAL avant de confirmer l'ecriture. |

---

## Decisions architecturales

| Decision | Justification |
|----------|--------------|
| 4KB pages par defaut | Aligne avec OS page, SSD sector |
| B+ tree pour entities | O(log n) lookup, excellent scan sequentiel |
| WAL pour crash recovery | Technique prouvee (SQLite) |
| mmap read-only | Zero-copy reads, pas de risque de corruption |
| CRC32 par page | Detection de corruption granulaire |
| Freelist chainee | Simple, pas d'overhead |

---

*Recherche effectuee le 2026-03-02.*
