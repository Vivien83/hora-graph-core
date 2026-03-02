use std::fmt;

/// All errors returned by hora-graph-core.
#[derive(Debug)]
pub enum HoraError {
    // Storage
    Io(std::io::Error),
    CorruptedFile {
        page: u32,
        expected_checksum: u32,
        actual_checksum: u32,
    },
    InvalidFile {
        reason: &'static str,
    },
    VersionMismatch {
        file_version: u16,
        min_supported: u16,
        max_supported: u16,
    },

    // Schema / Input
    EntityNotFound(u64),
    EdgeNotFound(u64),
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    AlreadyInvalidated(u64),

    // Capacity
    StringTooLong {
        max: usize,
        got: usize,
    },
    StorageFull,
}

impl fmt::Display for HoraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::CorruptedFile { page, expected_checksum, actual_checksum } => write!(
                f,
                "corrupted page {}: expected checksum 0x{:08x}, got 0x{:08x}",
                page, expected_checksum, actual_checksum
            ),
            Self::InvalidFile { reason } => write!(f, "invalid .hora file: {}", reason),
            Self::VersionMismatch { file_version, min_supported, max_supported } => write!(
                f,
                "version {} not supported (supported: {}-{})",
                file_version, min_supported, max_supported
            ),
            Self::EntityNotFound(id) => write!(f, "entity {} not found", id),
            Self::EdgeNotFound(id) => write!(f, "edge {} not found", id),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "embedding dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::AlreadyInvalidated(id) => write!(f, "fact {} is already invalidated", id),
            Self::StringTooLong { max, got } => {
                write!(f, "string too long: max {} bytes, got {}", max, got)
            }
            Self::StorageFull => write!(f, "storage is full"),
        }
    }
}

impl std::error::Error for HoraError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
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

pub type Result<T> = std::result::Result<T, HoraError>;
