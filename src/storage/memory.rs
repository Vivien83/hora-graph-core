use std::collections::HashMap;

use crate::core::edge::Edge;
use crate::core::entity::Entity;
use crate::core::episode::Episode;
use crate::core::types::{EdgeId, EntityId, StorageStats};
use crate::error::Result;
use crate::storage::traits::StorageOps;

/// In-memory storage backend. Used for tests and ephemeral instances.
pub struct MemoryStorage {
    entities: HashMap<EntityId, Entity>,
    edges: HashMap<EdgeId, Edge>,
    /// Maps an entity ID to the IDs of all edges where it appears as source or target.
    entity_edges: HashMap<EntityId, Vec<EdgeId>>,
    episodes: Vec<Episode>,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            edges: HashMap::new(),
            entity_edges: HashMap::new(),
            episodes: Vec::new(),
        }
    }
}

impl StorageOps for MemoryStorage {
    fn put_entity(&mut self, entity: Entity) -> Result<()> {
        self.entities.insert(entity.id, entity);
        Ok(())
    }

    fn get_entity(&self, id: EntityId) -> Result<Option<Entity>> {
        Ok(self.entities.get(&id).cloned())
    }

    fn delete_entity(&mut self, id: EntityId) -> Result<bool> {
        let existed = self.entities.remove(&id).is_some();
        self.entity_edges.remove(&id);
        Ok(existed)
    }

    fn put_edge(&mut self, edge: Edge) -> Result<()> {
        let id = edge.id;
        let source = edge.source;
        let target = edge.target;

        self.edges.insert(id, edge);
        self.entity_edges.entry(source).or_default().push(id);
        if source != target {
            self.entity_edges.entry(target).or_default().push(id);
        }
        Ok(())
    }

    fn get_edge(&self, id: EdgeId) -> Result<Option<Edge>> {
        Ok(self.edges.get(&id).cloned())
    }

    fn get_entity_edges(&self, entity_id: EntityId) -> Result<Vec<Edge>> {
        let edge_ids = match self.entity_edges.get(&entity_id) {
            Some(ids) => ids,
            None => return Ok(Vec::new()),
        };

        let edges = edge_ids
            .iter()
            .filter_map(|id| self.edges.get(id).cloned())
            .collect();
        Ok(edges)
    }

    fn get_entity_edge_ids(&self, entity_id: EntityId) -> Result<Vec<EdgeId>> {
        Ok(self
            .entity_edges
            .get(&entity_id)
            .cloned()
            .unwrap_or_default())
    }

    fn delete_edge(&mut self, id: EdgeId) -> Result<bool> {
        let edge = match self.edges.remove(&id) {
            Some(e) => e,
            None => return Ok(false),
        };

        // Clean up entity_edges index for both sides
        if let Some(ids) = self.entity_edges.get_mut(&edge.source) {
            ids.retain(|eid| *eid != id);
        }
        if edge.source != edge.target {
            if let Some(ids) = self.entity_edges.get_mut(&edge.target) {
                ids.retain(|eid| *eid != id);
            }
        }
        Ok(true)
    }

    fn scan_all_edges(&self) -> Result<Vec<Edge>> {
        Ok(self.edges.values().cloned().collect())
    }

    fn put_episode(&mut self, episode: Episode) -> Result<()> {
        self.episodes.push(episode);
        Ok(())
    }

    fn stats(&self) -> StorageStats {
        StorageStats {
            entities: self.entities.len() as u64,
            edges: self.edges.len() as u64,
            episodes: self.episodes.len() as u64,
        }
    }
}
