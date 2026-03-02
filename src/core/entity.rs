use crate::core::types::{EntityId, Properties};

/// A node in the knowledge graph.
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: EntityId,
    pub entity_type: String,
    pub name: String,
    pub properties: Properties,
    pub embedding: Option<Vec<f32>>,
    pub created_at: i64,
}
