use crate::core::types::{EdgeId, EntityId};

/// A directed edge (fact) between two entities in the knowledge graph.
///
/// Edges are bi-temporal: `valid_at` / `invalid_at` represent when the fact
/// is true in the real world, while `created_at` records when it was stored.
#[derive(Debug, Clone)]
pub struct Edge {
    pub id: EdgeId,
    pub source: EntityId,
    pub target: EntityId,
    pub relation_type: String,
    pub description: String,
    pub confidence: f32,
    pub valid_at: i64,
    pub invalid_at: i64,
    pub created_at: i64,
}
