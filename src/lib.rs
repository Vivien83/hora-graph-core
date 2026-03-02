pub mod core;
pub mod error;
pub mod storage;

pub use crate::core::edge::Edge;
pub use crate::core::entity::Entity;
pub use crate::core::episode::Episode;
pub use crate::core::types::{
    EdgeId, EntityId, EntityUpdate, EpisodeSource, FactUpdate, HoraConfig, Properties,
    PropertyValue, StorageStats,
};
pub use crate::error::{HoraError, Result};

use crate::core::types::now_millis;
use crate::storage::memory::MemoryStorage;
use crate::storage::traits::StorageOps;

/// The main entry point for hora-graph-core.
///
/// ```
/// use hora_graph_core::{HoraCore, HoraConfig};
///
/// let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
/// let id = hora.add_entity("project", "hora", None, None).unwrap();
/// let entity = hora.get_entity(id).unwrap().unwrap();
/// assert_eq!(entity.name, "hora");
/// ```
pub struct HoraCore {
    config: HoraConfig,
    storage: Box<dyn StorageOps>,
    next_entity_id: u64,
    next_edge_id: u64,
    next_episode_id: u64,
}

impl HoraCore {
    /// Create a new in-memory HoraCore instance.
    pub fn new(config: HoraConfig) -> Result<Self> {
        Ok(Self {
            config,
            storage: Box::new(MemoryStorage::new()),
            next_entity_id: 1,
            next_edge_id: 1,
            next_episode_id: 1,
        })
    }

    // --- CRUD Entities ---

    /// Add a new entity to the knowledge graph.
    ///
    /// Returns the ID of the newly created entity.
    pub fn add_entity(
        &mut self,
        entity_type: &str,
        name: &str,
        properties: Option<Properties>,
        embedding: Option<&[f32]>,
    ) -> Result<EntityId> {
        // Validate embedding dimensions
        if let Some(emb) = embedding {
            if self.config.embedding_dims == 0 {
                return Err(HoraError::DimensionMismatch {
                    expected: 0,
                    got: emb.len(),
                });
            }
            if emb.len() != self.config.embedding_dims as usize {
                return Err(HoraError::DimensionMismatch {
                    expected: self.config.embedding_dims as usize,
                    got: emb.len(),
                });
            }
        }

        let id = EntityId(self.next_entity_id);
        self.next_entity_id += 1;

        let entity = Entity {
            id,
            entity_type: entity_type.to_string(),
            name: name.to_string(),
            properties: properties.unwrap_or_default(),
            embedding: embedding.map(|e| e.to_vec()),
            created_at: now_millis(),
        };

        self.storage.put_entity(entity)?;
        Ok(id)
    }

    /// Get an entity by ID. Returns `None` if not found.
    pub fn get_entity(&self, id: EntityId) -> Result<Option<Entity>> {
        self.storage.get_entity(id)
    }

    /// Update an entity. Only fields set to `Some` in the update are changed.
    pub fn update_entity(&mut self, id: EntityId, update: EntityUpdate) -> Result<()> {
        let mut entity = self
            .storage
            .get_entity(id)?
            .ok_or(HoraError::EntityNotFound(id.0))?;

        if let Some(name) = update.name {
            entity.name = name;
        }
        if let Some(entity_type) = update.entity_type {
            entity.entity_type = entity_type;
        }
        if let Some(properties) = update.properties {
            entity.properties = properties;
        }
        if let Some(embedding) = update.embedding {
            if self.config.embedding_dims == 0 {
                return Err(HoraError::DimensionMismatch {
                    expected: 0,
                    got: embedding.len(),
                });
            }
            if embedding.len() != self.config.embedding_dims as usize {
                return Err(HoraError::DimensionMismatch {
                    expected: self.config.embedding_dims as usize,
                    got: embedding.len(),
                });
            }
            entity.embedding = Some(embedding);
        }

        self.storage.put_entity(entity)
    }

    /// Delete an entity and all its associated edges (cascade).
    pub fn delete_entity(&mut self, id: EntityId) -> Result<()> {
        if self.storage.get_entity(id)?.is_none() {
            return Err(HoraError::EntityNotFound(id.0));
        }

        // Cascade: delete all edges connected to this entity
        let edge_ids = self.storage.get_entity_edge_ids(id)?;
        for edge_id in edge_ids {
            self.storage.delete_edge(edge_id)?;
        }

        self.storage.delete_entity(id)?;
        Ok(())
    }

    // --- CRUD Facts (edges) ---

    /// Add a new fact (directed edge) between two entities.
    ///
    /// Returns the ID of the newly created fact. Both source and target
    /// entities must exist.
    pub fn add_fact(
        &mut self,
        source: EntityId,
        target: EntityId,
        relation: &str,
        description: &str,
        confidence: Option<f32>,
    ) -> Result<EdgeId> {
        // Verify both entities exist
        if self.storage.get_entity(source)?.is_none() {
            return Err(HoraError::EntityNotFound(source.0));
        }
        if self.storage.get_entity(target)?.is_none() {
            return Err(HoraError::EntityNotFound(target.0));
        }

        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        let now = now_millis();

        let edge = Edge {
            id,
            source,
            target,
            relation_type: relation.to_string(),
            description: description.to_string(),
            confidence: confidence.unwrap_or(1.0),
            valid_at: now,
            invalid_at: 0,
            created_at: now,
        };

        self.storage.put_edge(edge)?;
        Ok(id)
    }

    /// Get a fact by ID. Returns `None` if not found.
    pub fn get_fact(&self, id: EdgeId) -> Result<Option<Edge>> {
        self.storage.get_edge(id)
    }

    /// Update a fact. Only fields set to `Some` in the update are changed.
    pub fn update_fact(&mut self, id: EdgeId, update: FactUpdate) -> Result<()> {
        let mut edge = self
            .storage
            .get_edge(id)?
            .ok_or(HoraError::EdgeNotFound(id.0))?;

        if let Some(confidence) = update.confidence {
            edge.confidence = confidence;
        }
        if let Some(description) = update.description {
            edge.description = description;
        }

        self.storage.put_edge(edge)
    }

    /// Mark a fact as invalid (bi-temporal). The fact is NOT deleted —
    /// it remains queryable with its validity window.
    pub fn invalidate_fact(&mut self, id: EdgeId) -> Result<()> {
        let mut edge = self
            .storage
            .get_edge(id)?
            .ok_or(HoraError::EdgeNotFound(id.0))?;

        if edge.invalid_at != 0 {
            return Err(HoraError::AlreadyInvalidated(id.0));
        }

        edge.invalid_at = now_millis();
        self.storage.put_edge(edge)
    }

    /// Physically delete a fact. Use `invalidate_fact` for bi-temporal soft-delete.
    pub fn delete_fact(&mut self, id: EdgeId) -> Result<()> {
        if !self.storage.delete_edge(id)? {
            return Err(HoraError::EdgeNotFound(id.0));
        }
        Ok(())
    }

    /// Get all facts where the given entity is source or target.
    pub fn get_entity_facts(&self, entity_id: EntityId) -> Result<Vec<Edge>> {
        self.storage.get_entity_edges(entity_id)
    }

    // --- Episodes ---

    /// Record an episode (interaction snapshot).
    pub fn add_episode(
        &mut self,
        source: EpisodeSource,
        session_id: &str,
        entity_ids: &[EntityId],
        fact_ids: &[EdgeId],
    ) -> Result<u64> {
        let id = self.next_episode_id;
        self.next_episode_id += 1;

        let episode = Episode {
            id,
            source,
            session_id: session_id.to_string(),
            entity_ids: entity_ids.to_vec(),
            fact_ids: fact_ids.to_vec(),
            created_at: now_millis(),
            consolidation_count: 0,
        };

        self.storage.put_episode(episode)?;
        Ok(id)
    }

    // --- Stats ---

    /// Get summary statistics about the knowledge graph.
    pub fn stats(&self) -> Result<StorageStats> {
        Ok(self.storage.stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_creation() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let id = hora.add_entity("project", "hora", None, None).unwrap();
        let entity = hora.get_entity(id).unwrap().unwrap();
        assert_eq!(entity.name, "hora");
        assert_eq!(entity.entity_type, "project");
    }

    #[test]
    fn test_edge_creation() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("project", "hora", None, None).unwrap();
        let b = hora.add_entity("language", "Rust", None, None).unwrap();
        let _fact = hora.add_fact(a, b, "built_with", "hora is built with Rust", None).unwrap();
        let edges = hora.get_entity_facts(a).unwrap();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_entity_id_auto_increment() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let id1 = hora.add_entity("a", "first", None, None).unwrap();
        let id2 = hora.add_entity("b", "second", None, None).unwrap();
        assert_ne!(id1, id2);
        assert_eq!(id1.0 + 1, id2.0);
    }

    #[test]
    fn test_entity_not_found() {
        let hora = HoraCore::new(HoraConfig::default()).unwrap();
        let result = hora.get_entity(EntityId(999)).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_fact_references_valid_entities() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("project", "hora", None, None).unwrap();
        let result = hora.add_fact(a, EntityId(999), "rel", "desc", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_bidirectional_lookup() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("project", "hora", None, None).unwrap();
        let b = hora.add_entity("language", "Rust", None, None).unwrap();
        hora.add_fact(a, b, "built_with", "hora is built with Rust", None).unwrap();

        // Edge visible from both source and target
        let from_a = hora.get_entity_facts(a).unwrap();
        let from_b = hora.get_entity_facts(b).unwrap();
        assert_eq!(from_a.len(), 1);
        assert_eq!(from_b.len(), 1);
        assert_eq!(from_a[0].id, from_b[0].id);
    }

    #[test]
    fn test_edge_temporal_defaults() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("a", "x", None, None).unwrap();
        let b = hora.add_entity("b", "y", None, None).unwrap();
        let eid = hora.add_fact(a, b, "rel", "desc", None).unwrap();
        let edge = hora.get_fact(eid).unwrap().unwrap();

        assert!(edge.valid_at > 0);
        assert_eq!(edge.invalid_at, 0); // still valid
        assert!(edge.created_at > 0);
        assert_eq!(edge.confidence, 1.0);
    }

    #[test]
    fn test_embedding_dimension_mismatch() {
        let config = HoraConfig { embedding_dims: 4 };
        let mut hora = HoraCore::new(config).unwrap();
        let wrong_dims = vec![1.0, 2.0]; // 2 instead of 4
        let result = hora.add_entity("a", "x", None, Some(&wrong_dims));
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_when_dims_zero() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let emb = vec![1.0, 2.0, 3.0];
        let result = hora.add_entity("a", "x", None, Some(&emb));
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_correct_dims() {
        let config = HoraConfig { embedding_dims: 3 };
        let mut hora = HoraCore::new(config).unwrap();
        let emb = vec![1.0, 2.0, 3.0];
        let id = hora.add_entity("a", "x", None, Some(&emb)).unwrap();
        let entity = hora.get_entity(id).unwrap().unwrap();
        assert_eq!(entity.embedding.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_properties() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let mut props = Properties::new();
        props.insert("language".to_string(), PropertyValue::String("Rust".to_string()));
        props.insert("stars".to_string(), PropertyValue::Int(42));

        let id = hora.add_entity("project", "hora", Some(props), None).unwrap();
        let entity = hora.get_entity(id).unwrap().unwrap();
        assert_eq!(
            entity.properties.get("language"),
            Some(&PropertyValue::String("Rust".to_string()))
        );
        assert_eq!(entity.properties.get("stars"), Some(&PropertyValue::Int(42)));
    }

    #[test]
    fn test_episode_creation() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("project", "hora", None, None).unwrap();
        let b = hora.add_entity("language", "Rust", None, None).unwrap();
        let fact = hora.add_fact(a, b, "built_with", "desc", None).unwrap();

        let ep_id = hora
            .add_episode(EpisodeSource::Conversation, "sess-1", &[a, b], &[fact])
            .unwrap();
        assert_eq!(ep_id, 1);
    }

    #[test]
    fn test_stats() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("project", "hora", None, None).unwrap();
        let b = hora.add_entity("language", "Rust", None, None).unwrap();
        hora.add_fact(a, b, "built_with", "desc", None).unwrap();

        let stats = hora.stats().unwrap();
        assert_eq!(stats.entities, 2);
        assert_eq!(stats.edges, 1);
        assert_eq!(stats.episodes, 0);
    }

    #[test]
    fn test_entity_id_display() {
        let id = EntityId(42);
        assert_eq!(format!("{}", id), "entity:42");
    }

    #[test]
    fn test_edge_id_display() {
        let id = EdgeId(7);
        assert_eq!(format!("{}", id), "edge:7");
    }

    // --- v0.1b tests ---

    #[test]
    fn test_update_entity() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let id = hora.add_entity("project", "hora", None, None).unwrap();

        hora.update_entity(
            id,
            EntityUpdate {
                name: Some("hora-graph-core".to_string()),
                ..Default::default()
            },
        )
        .unwrap();

        let entity = hora.get_entity(id).unwrap().unwrap();
        assert_eq!(entity.name, "hora-graph-core");
        assert_eq!(entity.entity_type, "project"); // unchanged
    }

    #[test]
    fn test_update_entity_not_found() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let result = hora.update_entity(EntityId(999), EntityUpdate::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_entity_cascades() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("project", "hora", None, None).unwrap();
        let b = hora.add_entity("language", "Rust", None, None).unwrap();
        let fact_id = hora.add_fact(a, b, "built_with", "desc", None).unwrap();

        hora.delete_entity(a).unwrap();

        // Entity gone
        assert!(hora.get_entity(a).unwrap().is_none());
        // Edge cascade-deleted
        assert!(hora.get_fact(fact_id).unwrap().is_none());
        // Other entity untouched
        assert!(hora.get_entity(b).unwrap().is_some());
        // b's edge list is clean
        assert_eq!(hora.get_entity_facts(b).unwrap().len(), 0);
    }

    #[test]
    fn test_delete_entity_not_found() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let result = hora.delete_entity(EntityId(999));
        assert!(result.is_err());
    }

    #[test]
    fn test_invalidate_fact() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("a", "x", None, None).unwrap();
        let b = hora.add_entity("b", "y", None, None).unwrap();
        let fact_id = hora.add_fact(a, b, "rel", "desc", None).unwrap();

        hora.invalidate_fact(fact_id).unwrap();

        let fact = hora.get_fact(fact_id).unwrap().unwrap();
        assert!(fact.invalid_at > 0);
        // Fact still exists, just marked as invalid
    }

    #[test]
    fn test_invalidate_fact_twice_errors() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("a", "x", None, None).unwrap();
        let b = hora.add_entity("b", "y", None, None).unwrap();
        let fact_id = hora.add_fact(a, b, "rel", "desc", None).unwrap();

        hora.invalidate_fact(fact_id).unwrap();
        let result = hora.invalidate_fact(fact_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_fact() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("a", "x", None, None).unwrap();
        let b = hora.add_entity("b", "y", None, None).unwrap();
        let fact_id = hora.add_fact(a, b, "rel", "desc", None).unwrap();

        hora.delete_fact(fact_id).unwrap();
        assert!(hora.get_fact(fact_id).unwrap().is_none());
        // Edge lists are clean
        assert_eq!(hora.get_entity_facts(a).unwrap().len(), 0);
        assert_eq!(hora.get_entity_facts(b).unwrap().len(), 0);
    }

    #[test]
    fn test_delete_fact_not_found() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let result = hora.delete_fact(EdgeId(999));
        assert!(result.is_err());
    }

    #[test]
    fn test_update_fact() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("a", "x", None, None).unwrap();
        let b = hora.add_entity("b", "y", None, None).unwrap();
        let fact_id = hora.add_fact(a, b, "rel", "desc", Some(0.5)).unwrap();

        hora.update_fact(
            fact_id,
            FactUpdate {
                confidence: Some(0.95),
                ..Default::default()
            },
        )
        .unwrap();

        let fact = hora.get_fact(fact_id).unwrap().unwrap();
        assert_eq!(fact.confidence, 0.95);
        assert_eq!(fact.description, "desc"); // unchanged
    }

    #[test]
    fn test_props_macro() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let id = hora
            .add_entity(
                "project",
                "hora",
                Some(props! { "language" => "Rust", "stars" => 42 }),
                None,
            )
            .unwrap();

        let entity = hora.get_entity(id).unwrap().unwrap();
        assert_eq!(
            entity.properties.get("language"),
            Some(&PropertyValue::String("Rust".into()))
        );
        assert_eq!(entity.properties.get("stars"), Some(&PropertyValue::Int(42)));
    }

    #[test]
    fn test_stats_after_delete() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("a", "x", None, None).unwrap();
        let b = hora.add_entity("b", "y", None, None).unwrap();
        hora.add_fact(a, b, "rel", "desc", None).unwrap();

        assert_eq!(hora.stats().unwrap().entities, 2);
        assert_eq!(hora.stats().unwrap().edges, 1);

        hora.delete_entity(a).unwrap();

        assert_eq!(hora.stats().unwrap().entities, 1);
        assert_eq!(hora.stats().unwrap().edges, 0); // cascade
    }
}
