use crate::core::types::{EdgeId, EntityId, EpisodeSource};

/// An episodic memory record — a snapshot of an interaction or event.
///
/// Episodes are the raw, fast-write store (analogous to the hippocampus).
/// They are progressively transformed into semantic facts via the dream cycle.
#[derive(Debug, Clone)]
pub struct Episode {
    pub id: u64,
    pub source: EpisodeSource,
    pub session_id: String,
    pub entity_ids: Vec<EntityId>,
    pub fact_ids: Vec<EdgeId>,
    pub created_at: i64,
    pub consolidation_count: u32,
}
