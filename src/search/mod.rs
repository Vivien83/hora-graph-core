pub mod bm25;
pub mod vector;

use crate::core::types::EntityId;

/// A search result: entity ID + similarity score.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub entity_id: EntityId,
    pub score: f32,
}
