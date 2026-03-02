# 09 — Error Handling

> Zero dependency. `std::error::Error` manuel.
> Pas de thiserror, pas de anyhow. Chaque erreur est structuree et actionnable.

---

## Strategie

### Principes

1. **Pas de `unwrap()` en code de production** — `unwrap()` uniquement dans les tests
2. **Erreurs structurees** — chaque variante porte le contexte necessaire au debug
3. **Pas de string errors** — jamais de `Err("something went wrong".into())`
4. **Recoverable vs Fatal** — distinction claire entre les deux
5. **Pas de panic** — sauf violation d'invariant interne (bug)

### Categorisation

| Categorie | Recoverabilite | Action utilisateur | Exemples |
|-----------|---------------|-------------------|----------|
| **Input** | Recoverable | Corriger l'input | DimensionMismatch, EntityNotFound |
| **Storage** | Depends | Retry ou repair | IoError, WalCorrupted |
| **Capacity** | Recoverable | Reduire la charge | StorageFull, TooManyProperties |
| **Concurrency** | Recoverable | Retry | WriteLockTimeout |
| **Internal** | Fatal | Rapport de bug | Invariant violations (panic) |

---

## HoraError enum

```rust
use std::fmt;
use std::error::Error;

#[derive(Debug)]
pub enum HoraError {
    // === Storage ===
    /// Erreur I/O systeme (fichier, mmap, fsync)
    Io(std::io::Error),
    /// Le fichier .hora est corrompu (checksum mismatch)
    CorruptedFile {
        page: u32,
        expected_checksum: u32,
        actual_checksum: u32,
    },
    /// Le WAL est corrompu (ne peut pas etre rejoue)
    WalCorrupted {
        frame: u32,
        reason: &'static str,
    },
    /// Le fichier n'est pas un fichier .hora valide
    InvalidFile {
        reason: &'static str,
    },
    /// Version du fichier non supportee
    VersionMismatch {
        file_version: u16,
        min_supported: u16,
        max_supported: u16,
    },

    // === Schema / Input ===
    /// Entity demandee non trouvee
    EntityNotFound(u64),
    /// Edge demande non trouve
    EdgeNotFound(u64),
    /// Dimension de l'embedding incorrecte
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    /// Type d'entite inconnu
    InvalidEntityType {
        type_name_len: usize,
    },
    /// Type de relation inconnu
    InvalidRelationType {
        type_name_len: usize,
    },
    /// Edge reference une entite inexistante
    DanglingEdge {
        edge_id: u64,
        missing_entity: u64,
    },
    /// Le fait est deja invalide
    AlreadyInvalidated(u64),

    // === Capacity ===
    /// String trop longue pour le string pool
    StringTooLong {
        max: usize,
        got: usize,
    },
    /// Trop de proprietes sur une entite
    TooManyProperties {
        max: usize,
        got: usize,
    },
    /// Le storage est plein (disk full ou limite configuree)
    StorageFull,
    /// Trop de types d'entites (max 255)
    TooManyEntityTypes {
        max: usize,
    },
    /// Trop de types de relations (max 255)
    TooManyRelationTypes {
        max: usize,
    },

    // === Concurrency ===
    /// Timeout en attendant le write lock
    WriteLockTimeout,
    /// Conflit de transaction (future v0.5+)
    TransactionConflict,
}

impl fmt::Display for HoraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::CorruptedFile { page, expected_checksum, actual_checksum } =>
                write!(f, "corrupted page {}: expected checksum 0x{:08x}, got 0x{:08x}",
                    page, expected_checksum, actual_checksum),
            Self::WalCorrupted { frame, reason } =>
                write!(f, "WAL corrupted at frame {}: {}", frame, reason),
            Self::InvalidFile { reason } =>
                write!(f, "invalid .hora file: {}", reason),
            Self::VersionMismatch { file_version, min_supported, max_supported } =>
                write!(f, "version {} not supported (supported: {}-{})",
                    file_version, min_supported, max_supported),
            Self::EntityNotFound(id) =>
                write!(f, "entity {} not found", id),
            Self::EdgeNotFound(id) =>
                write!(f, "edge {} not found", id),
            Self::DimensionMismatch { expected, got } =>
                write!(f, "embedding dimension mismatch: expected {}, got {}", expected, got),
            Self::InvalidEntityType { type_name_len } =>
                write!(f, "invalid entity type (name length: {})", type_name_len),
            Self::InvalidRelationType { type_name_len } =>
                write!(f, "invalid relation type (name length: {})", type_name_len),
            Self::DanglingEdge { edge_id, missing_entity } =>
                write!(f, "edge {} references missing entity {}", edge_id, missing_entity),
            Self::AlreadyInvalidated(id) =>
                write!(f, "fact {} is already invalidated", id),
            Self::StringTooLong { max, got } =>
                write!(f, "string too long: max {} bytes, got {}", max, got),
            Self::TooManyProperties { max, got } =>
                write!(f, "too many properties: max {}, got {}", max, got),
            Self::StorageFull =>
                write!(f, "storage is full"),
            Self::TooManyEntityTypes { max } =>
                write!(f, "too many entity types (max {})", max),
            Self::TooManyRelationTypes { max } =>
                write!(f, "too many relation types (max {})", max),
            Self::WriteLockTimeout =>
                write!(f, "write lock timeout"),
            Self::TransactionConflict =>
                write!(f, "transaction conflict"),
        }
    }
}

impl Error for HoraError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for HoraError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
```

---

## Result type alias

```rust
pub type Result<T> = std::result::Result<T, HoraError>;
```

Usage :
```rust
pub fn get_entity(&self, id: EntityId) -> Result<Option<Entity>> { ... }
pub fn add_entity(&self, ...) -> Result<EntityId> { ... }
```

---

## Error mapping pour bindings

### Node.js (napi-rs)

napi-rs convertit automatiquement les `Result<T>` en exceptions JS.
Le type d'erreur est expose via le message.

```rust
#[napi]
impl HoraCore {
    #[napi]
    pub fn get_entity(&self, id: i64) -> napi::Result<Option<EntityJs>> {
        self.inner.get_entity(EntityId(id as u64))
            .map(|opt| opt.map(EntityJs::from))
            .map_err(|e| napi::Error::from_reason(format!("{}", e)))
    }
}
```

### TypeScript error types

```typescript
// Auto-genere par napi-rs
class HoraError extends Error {
  // Le message contient le detail structure
  // Pattern matching via prefix :
  //   "entity 42 not found"
  //   "embedding dimension mismatch: expected 384, got 768"
  //   "corrupted page 5: expected checksum 0x..."
}
```

Pour un mapping plus riche en v0.5+ :
```typescript
type HoraErrorType =
  | { type: 'DimensionMismatch'; expected: number; got: number }
  | { type: 'EntityNotFound'; id: bigint }
  | { type: 'EdgeNotFound'; id: bigint }
  | { type: 'StorageFull' }
  | { type: 'CorruptedFile'; page: number }
  | { type: 'IoError'; message: string };
```

### Python (PyO3)

```rust
#[pymethods]
impl HoraCore {
    fn get_entity(&self, id: u64) -> PyResult<Option<EntityPy>> {
        self.inner.get_entity(EntityId(id))
            .map(|opt| opt.map(EntityPy::from))
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }
}
```

---

## Invariant violations (panics)

Les panics sont reservees aux violations d'invariants internes — des bugs dans hora.

```rust
// OK : panic sur invariant interne
debug_assert!(page_num < self.page_count, "page number out of bounds");

// JAMAIS : panic sur input utilisateur
// BAD: assert!(embedding.len() == self.dims);
// GOOD: if embedding.len() != self.dims { return Err(DimensionMismatch {...}); }
```

### Liste des invariants qui peuvent paniquer

| Invariant | Quand | Cause probable |
|-----------|-------|---------------|
| `entity_id < next_entity_id` | Lookup par ID | Bug dans l'allocateur d'ID |
| `page_num < page_count` | Lecture de page | Bug dans le B+ tree |
| `offset < string_pool_len` | Lecture de string | Bug dans le string pool |
| `recent_count <= 10` | Activation record | Bug dans record_access |

---

## CRC32 Implementation (zero-dep)

```rust
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    crc ^ 0xFFFF_FFFF
}
```

---

## Decisions prises

| Decision | Justification |
|----------|--------------|
| Pas de thiserror | Zero dep. ~80 lignes de boilerplate acceptable |
| Pas de error codes numeriques | Les strings structurees suffisent pour le debug |
| `Option<Entity>` pour get | Un entity non trouvee n'est pas une erreur |
| Panic sur invariants internes | Distingue bug interne vs erreur utilisateur |
| CRC32 maison | ~50 lignes, evite crc32fast dependency |

---

*Document cree le 2026-03-02.*
