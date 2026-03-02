use std::collections::HashMap;
use std::fmt;

/// Unique identifier for an entity in the knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityId(pub u64);

/// Unique identifier for an edge (fact) in the knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub u64);

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "entity:{}", self.0)
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "edge:{}", self.0)
    }
}

/// A property value attached to an entity.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    String(std::string::String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// Convenience type alias for entity properties.
pub type Properties = HashMap<std::string::String, PropertyValue>;

/// Source of an episode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodeSource {
    Conversation,
    Document,
    Api,
}

/// Configuration for a HoraCore instance.
#[derive(Debug, Clone, Default)]
pub struct HoraConfig {
    /// Embedding vector dimensions. 0 = no vector search (text-only mode).
    pub embedding_dims: u16,
}

/// Summary statistics for the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageStats {
    pub entities: u64,
    pub edges: u64,
    pub episodes: u64,
}

/// Current time in epoch milliseconds.
pub(crate) fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_millis() as i64
}
