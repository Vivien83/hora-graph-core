use crate::core::edge::Edge;
use crate::core::entity::Entity;
use crate::core::episode::Episode;
use crate::core::types::{EdgeId, EntityId, StorageStats};
use crate::error::Result;

/// Abstraction over storage backends (memory, embedded, sqlite, postgres).
///
/// All backends implement the same trait so the engine behaves identically
/// regardless of the underlying storage.
pub trait StorageOps {
    // --- Entities ---
    fn put_entity(&mut self, entity: Entity) -> Result<()>;
    fn get_entity(&self, id: EntityId) -> Result<Option<Entity>>;

    // --- Edges ---
    fn put_edge(&mut self, edge: Edge) -> Result<()>;
    fn get_edge(&self, id: EdgeId) -> Result<Option<Edge>>;
    fn get_entity_edges(&self, entity_id: EntityId) -> Result<Vec<Edge>>;

    // --- Episodes ---
    fn put_episode(&mut self, episode: Episode) -> Result<()>;

    // --- Stats ---
    fn stats(&self) -> StorageStats;
}
