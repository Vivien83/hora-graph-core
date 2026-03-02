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
    fn delete_entity(&mut self, id: EntityId) -> Result<bool>;

    // --- Edges ---
    fn put_edge(&mut self, edge: Edge) -> Result<()>;
    fn get_edge(&self, id: EdgeId) -> Result<Option<Edge>>;
    fn get_entity_edges(&self, entity_id: EntityId) -> Result<Vec<Edge>>;
    fn get_entity_edge_ids(&self, entity_id: EntityId) -> Result<Vec<EdgeId>>;
    fn delete_edge(&mut self, id: EdgeId) -> Result<bool>;

    // --- Episodes ---
    fn put_episode(&mut self, episode: Episode) -> Result<()>;

    // --- Scan ---
    /// Return all entities in storage.
    fn scan_all_entities(&self) -> Result<Vec<Entity>>;
    /// Return all edges in storage.
    fn scan_all_edges(&self) -> Result<Vec<Edge>>;
    /// Return all episodes in storage.
    fn scan_all_episodes(&self) -> Result<Vec<Episode>>;

    // --- Stats ---
    fn stats(&self) -> StorageStats;
}
