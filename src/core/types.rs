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

impl From<&str> for PropertyValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<std::string::String> for PropertyValue {
    fn from(s: std::string::String) -> Self {
        Self::String(s)
    }
}

impl From<i64> for PropertyValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<i32> for PropertyValue {
    fn from(v: i32) -> Self {
        Self::Int(v as i64)
    }
}

impl From<f64> for PropertyValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<bool> for PropertyValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

/// Convenience type alias for entity properties.
pub type Properties = HashMap<std::string::String, PropertyValue>;

/// Convenience macro for building property maps.
///
/// ```
/// use hora_graph_core::{props, PropertyValue};
/// let p = props! { "language" => "Rust", "stars" => 42 };
/// assert_eq!(p.get("language"), Some(&PropertyValue::String("Rust".into())));
/// ```
#[macro_export]
macro_rules! props {
    ($($key:expr => $val:expr),* $(,)?) => {{
        let mut map = $crate::Properties::new();
        $(map.insert($key.to_string(), $crate::PropertyValue::from($val));)*
        map
    }};
}

/// Partial update for an entity. Only `Some` fields are applied.
#[derive(Debug, Clone, Default)]
pub struct EntityUpdate {
    pub name: Option<std::string::String>,
    pub entity_type: Option<std::string::String>,
    pub properties: Option<Properties>,
    pub embedding: Option<Vec<f32>>,
}

/// Partial update for a fact. Only `Some` fields are applied.
#[derive(Debug, Clone, Default)]
pub struct FactUpdate {
    pub confidence: Option<f32>,
    pub description: Option<std::string::String>,
}

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
