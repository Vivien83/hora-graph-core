pub mod core;
pub mod error;
pub mod search;
pub mod storage;

pub use crate::core::edge::Edge;
pub use crate::core::entity::Entity;
pub use crate::core::episode::Episode;
pub use crate::core::types::{
    EdgeId, EntityId, EntityUpdate, EpisodeSource, FactUpdate, HoraConfig, Properties,
    PropertyValue, StorageStats, TraverseOpts, TraverseResult,
};
pub use crate::error::{HoraError, Result};
pub use crate::search::SearchHit;

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};

use crate::core::types::now_millis;
use crate::search::bm25::{self, Bm25Index};
use crate::storage::format::{self, FileHeader};
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
    file_path: Option<PathBuf>,
    bm25_index: Bm25Index,
}

impl HoraCore {
    /// Create a new in-memory HoraCore instance (no persistence).
    pub fn new(config: HoraConfig) -> Result<Self> {
        Ok(Self {
            config,
            storage: Box::new(MemoryStorage::new()),
            next_entity_id: 1,
            next_edge_id: 1,
            next_episode_id: 1,
            file_path: None,
            bm25_index: Bm25Index::new(),
        })
    }

    /// Open a file-backed HoraCore instance.
    ///
    /// If the file exists, loads data from it. If it does not exist, creates
    /// a new empty instance that will write to the given path on `flush()`.
    pub fn open(path: impl AsRef<Path>, config: HoraConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if path.exists() {
            let file = std::fs::File::open(&path)?;
            let mut reader = std::io::BufReader::new(file);
            let graph = format::deserialize(&mut reader)?;

            // Rebuild MemoryStorage and BM25 index from deserialized data
            let mut storage = MemoryStorage::new();
            let mut bm25 = Bm25Index::new();
            for entity in graph.entities {
                let text = bm25::entity_text(&entity.name, &entity.properties);
                bm25.index_document(entity.id.0 as u32, &text);
                storage.put_entity(entity)?;
            }
            for edge in graph.edges {
                storage.put_edge(edge)?;
            }
            for episode in graph.episodes {
                storage.put_episode(episode)?;
            }

            Ok(Self {
                config: HoraConfig {
                    embedding_dims: graph.header.embedding_dims,
                },
                storage: Box::new(storage),
                next_entity_id: graph.header.next_entity_id,
                next_edge_id: graph.header.next_edge_id,
                next_episode_id: graph.header.next_episode_id,
                file_path: Some(path),
                bm25_index: bm25,
            })
        } else {
            Ok(Self {
                file_path: Some(path),
                ..Self::new(config)?
            })
        }
    }

    // --- Persistence ---

    /// Flush all data to the backing file.
    ///
    /// Writes to a temporary file first, then renames for crash safety.
    /// Returns an error if this is an in-memory-only instance.
    pub fn flush(&self) -> Result<()> {
        let path = self
            .file_path
            .as_ref()
            .ok_or(HoraError::InvalidFile {
                reason: "cannot flush an in-memory-only instance",
            })?;

        let entities = self.storage.scan_all_entities()?;
        let edges = self.storage.scan_all_edges()?;
        let episodes = self.storage.scan_all_episodes()?;

        let header = FileHeader {
            embedding_dims: self.config.embedding_dims,
            next_entity_id: self.next_entity_id,
            next_edge_id: self.next_edge_id,
            next_episode_id: self.next_episode_id,
            entity_count: entities.len() as u32,
            edge_count: edges.len() as u32,
            episode_count: episodes.len() as u32,
        };

        // Write to .tmp then rename (crash-safe)
        let tmp_path = path.with_extension("hora.tmp");
        {
            let file = std::fs::File::create(&tmp_path)?;
            let mut writer = std::io::BufWriter::new(file);
            format::serialize(&mut writer, &header, &entities, &edges, &episodes)?;
        }
        std::fs::rename(&tmp_path, path)?;

        Ok(())
    }

    /// Copy the current state to a snapshot file.
    ///
    /// Flushes first if file-backed, then copies. For in-memory instances,
    /// writes directly to the given path.
    pub fn snapshot(&self, dest: impl AsRef<Path>) -> Result<()> {
        let entities = self.storage.scan_all_entities()?;
        let edges = self.storage.scan_all_edges()?;
        let episodes = self.storage.scan_all_episodes()?;

        let header = FileHeader {
            embedding_dims: self.config.embedding_dims,
            next_entity_id: self.next_entity_id,
            next_edge_id: self.next_edge_id,
            next_episode_id: self.next_episode_id,
            entity_count: entities.len() as u32,
            edge_count: edges.len() as u32,
            episode_count: episodes.len() as u32,
        };

        let file = std::fs::File::create(dest)?;
        let mut writer = std::io::BufWriter::new(file);
        format::serialize(&mut writer, &header, &entities, &edges, &episodes)?;

        Ok(())
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

        // Index for BM25 full-text search
        let text = bm25::entity_text(&entity.name, &entity.properties);
        self.bm25_index.index_document(id.0 as u32, &text);

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

        // Re-index for BM25
        let text = bm25::entity_text(&entity.name, &entity.properties);
        self.bm25_index.index_document(id.0 as u32, &text);

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
        self.bm25_index.remove_document(id.0 as u32);
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

    // --- Graph Traversal ---

    /// BFS traversal from a start entity up to the given depth.
    ///
    /// Returns IDs of all discovered entities and edges.
    /// Depth 0 returns only the start entity (no edges).
    pub fn traverse(&self, start: EntityId, opts: TraverseOpts) -> Result<TraverseResult> {
        if self.storage.get_entity(start)?.is_none() {
            return Err(HoraError::EntityNotFound(start.0));
        }

        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut result_entity_ids = vec![start];
        let mut result_edge_ids: Vec<EdgeId> = Vec::new();
        let mut seen_edges: HashSet<EdgeId> = HashSet::new();

        visited.insert(start);

        // BFS queue holds (entity_id, current_depth)
        let mut queue: VecDeque<(EntityId, u32)> = VecDeque::new();
        queue.push_back((start, 0));

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= opts.depth {
                continue;
            }

            let edges = self.storage.get_entity_edges(current_id)?;
            for edge in edges {
                if !seen_edges.insert(edge.id) {
                    continue;
                }

                // The neighbor is whichever end is NOT current_id
                let neighbor_id = if edge.source == current_id {
                    edge.target
                } else {
                    edge.source
                };

                result_edge_ids.push(edge.id);

                if visited.insert(neighbor_id)
                    && self.storage.get_entity(neighbor_id)?.is_some()
                {
                    result_entity_ids.push(neighbor_id);
                    queue.push_back((neighbor_id, depth + 1));
                }
            }
        }

        Ok(TraverseResult {
            entity_ids: result_entity_ids,
            edge_ids: result_edge_ids,
        })
    }

    /// Get all direct neighbor entity IDs (connected via any edge).
    pub fn neighbors(&self, entity_id: EntityId) -> Result<Vec<EntityId>> {
        let edges = self.storage.get_entity_edges(entity_id)?;
        let mut neighbor_ids: HashSet<EntityId> = HashSet::new();

        for edge in &edges {
            if edge.source == entity_id {
                neighbor_ids.insert(edge.target);
            } else {
                neighbor_ids.insert(edge.source);
            }
        }

        // Remove self (possible with self-loops)
        neighbor_ids.remove(&entity_id);
        Ok(neighbor_ids.into_iter().collect())
    }

    /// Timeline of all facts involving an entity, sorted by `valid_at`.
    pub fn timeline(&self, entity_id: EntityId) -> Result<Vec<Edge>> {
        let mut edges = self.storage.get_entity_edges(entity_id)?;
        edges.sort_by_key(|e| e.valid_at);
        Ok(edges)
    }

    /// All facts valid at a given point in time.
    ///
    /// A fact is valid at time `t` if `valid_at <= t` and
    /// (`invalid_at == 0` or `invalid_at > t`).
    pub fn facts_at(&self, t: i64) -> Result<Vec<Edge>> {
        let all = self.storage.scan_all_edges()?;
        let valid: Vec<Edge> = all
            .into_iter()
            .filter(|e| e.valid_at <= t && (e.invalid_at == 0 || e.invalid_at > t))
            .collect();
        Ok(valid)
    }

    // --- Vector Search ---

    /// Brute-force vector search: find the `k` most similar entities by cosine similarity.
    ///
    /// Requires `embedding_dims > 0` in the config. Entities without embeddings
    /// are silently skipped. The query embedding must match `embedding_dims` in length.
    pub fn vector_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        if self.config.embedding_dims == 0 {
            return Err(HoraError::DimensionMismatch {
                expected: 0,
                got: query.len(),
            });
        }
        if query.len() != self.config.embedding_dims as usize {
            return Err(HoraError::DimensionMismatch {
                expected: self.config.embedding_dims as usize,
                got: query.len(),
            });
        }

        let entities = self.storage.scan_all_entities()?;

        // Collect (id, embedding_slice) pairs, skip entities without embeddings
        let with_embeddings: Vec<(EntityId, &[f32])> = entities
            .iter()
            .filter_map(|e| e.embedding.as_ref().map(|emb| (e.id, emb.as_slice())))
            .collect();

        Ok(search::vector::top_k_brute_force(query, &with_embeddings, k))
    }

    // --- Text Search (BM25) ---

    /// Full-text search using BM25+ scoring over entity names and string properties.
    ///
    /// Returns the top `k` matching entities. Entities without indexable text
    /// are invisible to BM25.
    pub fn text_search(&mut self, query: &str, k: usize) -> Result<Vec<SearchHit>> {
        Ok(self.bm25_index.search(query, k))
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

    // --- v0.1c tests: Graph Traversal ---

    #[test]
    fn test_bfs_depth_2() {
        // A -> B -> C -> D
        // traverse(A, depth=2) should return {A, B, C} but not D
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("node", "A", None, None).unwrap();
        let b = hora.add_entity("node", "B", None, None).unwrap();
        let c = hora.add_entity("node", "C", None, None).unwrap();
        let d = hora.add_entity("node", "D", None, None).unwrap();

        hora.add_fact(a, b, "link", "A->B", None).unwrap();
        hora.add_fact(b, c, "link", "B->C", None).unwrap();
        hora.add_fact(c, d, "link", "C->D", None).unwrap();

        let result = hora.traverse(a, TraverseOpts { depth: 2 }).unwrap();

        assert!(result.entity_ids.contains(&a));
        assert!(result.entity_ids.contains(&b));
        assert!(result.entity_ids.contains(&c));
        assert!(!result.entity_ids.contains(&d));
        assert_eq!(result.entity_ids.len(), 3);
        assert_eq!(result.edge_ids.len(), 2); // A->B and B->C
    }

    #[test]
    fn test_bfs_depth_0() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("node", "A", None, None).unwrap();
        let b = hora.add_entity("node", "B", None, None).unwrap();
        hora.add_fact(a, b, "link", "A->B", None).unwrap();

        let result = hora.traverse(a, TraverseOpts { depth: 0 }).unwrap();
        assert_eq!(result.entity_ids.len(), 1);
        assert_eq!(result.entity_ids[0], a);
        assert_eq!(result.edge_ids.len(), 0);
    }

    #[test]
    fn test_bfs_cycle() {
        // A -> B -> C -> A (cycle)
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("node", "A", None, None).unwrap();
        let b = hora.add_entity("node", "B", None, None).unwrap();
        let c = hora.add_entity("node", "C", None, None).unwrap();

        hora.add_fact(a, b, "link", "A->B", None).unwrap();
        hora.add_fact(b, c, "link", "B->C", None).unwrap();
        hora.add_fact(c, a, "link", "C->A", None).unwrap();

        // Should not infinite loop, and should find all 3 nodes
        let result = hora.traverse(a, TraverseOpts { depth: 10 }).unwrap();
        assert_eq!(result.entity_ids.len(), 3);
        assert_eq!(result.edge_ids.len(), 3);
    }

    #[test]
    fn test_bfs_isolated_node() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("node", "lonely", None, None).unwrap();

        let result = hora.traverse(a, TraverseOpts { depth: 5 }).unwrap();
        assert_eq!(result.entity_ids.len(), 1);
        assert_eq!(result.edge_ids.len(), 0);
    }

    #[test]
    fn test_bfs_not_found() {
        let hora = HoraCore::new(HoraConfig::default()).unwrap();
        let result = hora.traverse(EntityId(999), TraverseOpts::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_neighbors() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("node", "A", None, None).unwrap();
        let b = hora.add_entity("node", "B", None, None).unwrap();
        let c = hora.add_entity("node", "C", None, None).unwrap();
        let d = hora.add_entity("node", "D", None, None).unwrap();

        hora.add_fact(a, b, "link", "A->B", None).unwrap();
        hora.add_fact(a, c, "link", "A->C", None).unwrap();
        // D is not connected to A

        let mut neighbors = hora.neighbors(a).unwrap();
        neighbors.sort();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&b));
        assert!(neighbors.contains(&c));
        assert!(!neighbors.contains(&d));
    }

    #[test]
    fn test_timeline_ordered() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("person", "Alice", None, None).unwrap();
        let b = hora.add_entity("company", "Acme", None, None).unwrap();
        let c = hora.add_entity("company", "BigCorp", None, None).unwrap();

        // Create facts — since they're created sequentially, valid_at increases
        let f1 = hora.add_fact(a, b, "works_at", "Alice at Acme", None).unwrap();
        let f2 = hora.add_fact(a, c, "works_at", "Alice at BigCorp", None).unwrap();

        let tl = hora.timeline(a).unwrap();
        assert_eq!(tl.len(), 2);
        assert_eq!(tl[0].id, f1);
        assert_eq!(tl[1].id, f2);
        assert!(tl[0].valid_at <= tl[1].valid_at);
    }

    #[test]
    fn test_facts_at_bitemporal() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("a", "x", None, None).unwrap();
        let b = hora.add_entity("b", "y", None, None).unwrap();

        // Manually craft edges with controlled timestamps via add_fact + update
        let f1 = hora.add_fact(a, b, "rel", "fact1", None).unwrap();
        let f2 = hora.add_fact(a, b, "rel2", "fact2", None).unwrap();

        // Get the actual timestamps so we can reason about them
        let e1 = hora.get_fact(f1).unwrap().unwrap();
        let e2 = hora.get_fact(f2).unwrap().unwrap();

        // Invalidate f1 — sets invalid_at to now
        hora.invalidate_fact(f1).unwrap();
        let e1_after = hora.get_fact(f1).unwrap().unwrap();

        // facts_at(before everything) = nothing
        let before = hora.facts_at(e1.valid_at - 1).unwrap();
        assert_eq!(before.len(), 0);

        // facts_at(between creation and invalidation) = both
        // Since f1 was valid from e1.valid_at until e1_after.invalid_at,
        // and f2 was valid from e2.valid_at with no end,
        // at time e2.valid_at both should be visible (before invalidation timestamp)
        let mid = hora.facts_at(e2.valid_at).unwrap();
        // f1 is still valid here (valid_at <= t, invalid_at > t because invalidation happens after)
        assert!(mid.iter().any(|e| e.id == f2));

        // facts_at(well into the future) = only f2 (f1 is invalidated)
        let future = hora.facts_at(e1_after.invalid_at + 1000).unwrap();
        assert!(future.iter().any(|e| e.id == f2));
        assert!(!future.iter().any(|e| e.id == f1));
    }

    #[test]
    fn test_facts_at_never_invalidated() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let a = hora.add_entity("a", "x", None, None).unwrap();
        let b = hora.add_entity("b", "y", None, None).unwrap();
        let f = hora.add_fact(a, b, "rel", "always valid", None).unwrap();

        let edge = hora.get_fact(f).unwrap().unwrap();

        // A fact with invalid_at=0 is valid at any time >= valid_at
        let result = hora.facts_at(edge.valid_at + 1_000_000).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, f);
    }

    // --- v0.1d tests: Persistence ---

    #[test]
    fn test_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hora");

        let (a_id, b_id, fact_id);
        {
            let mut hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            a_id = hora.add_entity("project", "hora", None, None).unwrap();
            b_id = hora
                .add_entity(
                    "language",
                    "Rust",
                    Some(props! { "year" => 2015 }),
                    None,
                )
                .unwrap();
            fact_id = hora
                .add_fact(a_id, b_id, "built_with", "hora uses Rust", Some(0.95))
                .unwrap();
            hora.flush().unwrap();
        }

        // Reopen and verify
        {
            let hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            let stats = hora.stats().unwrap();
            assert_eq!(stats.entities, 2);
            assert_eq!(stats.edges, 1);

            let a = hora.get_entity(a_id).unwrap().unwrap();
            assert_eq!(a.name, "hora");
            assert_eq!(a.entity_type, "project");

            let b = hora.get_entity(b_id).unwrap().unwrap();
            assert_eq!(b.name, "Rust");
            assert_eq!(
                b.properties.get("year"),
                Some(&PropertyValue::Int(2015))
            );

            let fact = hora.get_fact(fact_id).unwrap().unwrap();
            assert_eq!(fact.relation_type, "built_with");
            assert_eq!(fact.confidence, 0.95);
        }
    }

    #[test]
    fn test_persistence_ids_continue() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hora");

        {
            let mut hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            hora.add_entity("a", "first", None, None).unwrap(); // id=1
            hora.add_entity("b", "second", None, None).unwrap(); // id=2
            hora.flush().unwrap();
        }

        {
            let mut hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            let id = hora.add_entity("c", "third", None, None).unwrap();
            // ID should continue from where we left off (3), not restart at 1
            assert_eq!(id.0, 3);
        }
    }

    #[test]
    fn test_persistence_with_embeddings() {
        let config = HoraConfig { embedding_dims: 3 };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hora");

        {
            let mut hora = HoraCore::open(&path, config.clone()).unwrap();
            let emb = vec![1.0, 2.0, 3.0];
            hora.add_entity("a", "x", None, Some(&emb)).unwrap();
            hora.flush().unwrap();
        }

        {
            let hora = HoraCore::open(&path, config).unwrap();
            let e = hora.get_entity(EntityId(1)).unwrap().unwrap();
            assert_eq!(e.embedding.as_ref().unwrap(), &[1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_persistence_with_episodes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hora");

        {
            let mut hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            let a = hora.add_entity("a", "x", None, None).unwrap();
            hora.add_episode(EpisodeSource::Conversation, "sess-1", &[a], &[])
                .unwrap();
            hora.flush().unwrap();
        }

        {
            let hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            let stats = hora.stats().unwrap();
            assert_eq!(stats.episodes, 1);
        }
    }

    #[test]
    fn test_persistence_invalidated_fact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hora");

        let fact_id;
        {
            let mut hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            let a = hora.add_entity("a", "x", None, None).unwrap();
            let b = hora.add_entity("b", "y", None, None).unwrap();
            fact_id = hora.add_fact(a, b, "rel", "desc", None).unwrap();
            hora.invalidate_fact(fact_id).unwrap();
            hora.flush().unwrap();
        }

        {
            let hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            let fact = hora.get_fact(fact_id).unwrap().unwrap();
            assert!(fact.invalid_at > 0);
        }
    }

    #[test]
    fn test_corrupted_file_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.hora");
        std::fs::write(&path, b"NOT_HORA_FILE").unwrap();

        let result = HoraCore::open(&path, HoraConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hora");
        let snap = dir.path().join("snapshot.hora");

        {
            let mut hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            hora.add_entity("project", "hora", None, None).unwrap();
            hora.flush().unwrap();
            hora.snapshot(&snap).unwrap();
        }

        // Open from snapshot
        {
            let hora = HoraCore::open(&snap, HoraConfig::default()).unwrap();
            assert_eq!(hora.stats().unwrap().entities, 1);
        }
    }

    #[test]
    fn test_flush_memory_only_errors() {
        let hora = HoraCore::new(HoraConfig::default()).unwrap();
        let result = hora.flush();
        assert!(result.is_err());
    }

    #[test]
    fn test_snapshot_memory_instance() {
        let dir = tempfile::tempdir().unwrap();
        let snap = dir.path().join("snapshot.hora");

        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        hora.add_entity("project", "hora", None, None).unwrap();
        hora.snapshot(&snap).unwrap();

        let hora2 = HoraCore::open(&snap, HoraConfig::default()).unwrap();
        assert_eq!(hora2.stats().unwrap().entities, 1);
    }

    #[test]
    fn test_persistence_all_property_types() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hora");

        {
            let mut hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            hora.add_entity(
                "test",
                "props",
                Some(props! {
                    "name" => "hora",
                    "stars" => 42,
                    "score" => 2.72,
                    "active" => true
                }),
                None,
            )
            .unwrap();
            hora.flush().unwrap();
        }

        {
            let hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            let e = hora.get_entity(EntityId(1)).unwrap().unwrap();
            assert_eq!(
                e.properties.get("name"),
                Some(&PropertyValue::String("hora".into()))
            );
            assert_eq!(e.properties.get("stars"), Some(&PropertyValue::Int(42)));
            assert_eq!(
                e.properties.get("score"),
                Some(&PropertyValue::Float(2.72))
            );
            assert_eq!(
                e.properties.get("active"),
                Some(&PropertyValue::Bool(true))
            );
        }
    }

    // --- v0.2a tests: Vector Search ---

    #[test]
    fn test_vector_search_basic() {
        let config = HoraConfig { embedding_dims: 3 };
        let mut hora = HoraCore::new(config).unwrap();

        // Entity close to query
        hora.add_entity("a", "close", None, Some(&[1.0, 0.0, 0.0]))
            .unwrap();
        // Entity far from query
        hora.add_entity("b", "far", None, Some(&[0.0, 1.0, 0.0]))
            .unwrap();
        // Entity very close to query
        hora.add_entity("c", "very_close", None, Some(&[0.9, 0.1, 0.0]))
            .unwrap();

        let results = hora.vector_search(&[1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        // First result should be the exact match
        assert_eq!(results[0].entity_id, EntityId(1));
        // Second should be the close one
        assert_eq!(results[1].entity_id, EntityId(3));
    }

    #[test]
    fn test_vector_search_returns_exact_k() {
        let config = HoraConfig { embedding_dims: 3 };
        let mut hora = HoraCore::new(config).unwrap();

        for i in 0..20 {
            let emb = vec![i as f32, 0.0, 1.0];
            hora.add_entity("node", &format!("n{}", i), None, Some(&emb))
                .unwrap();
        }

        let results = hora.vector_search(&[10.0, 0.0, 1.0], 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_vector_search_skips_no_embedding() {
        let config = HoraConfig { embedding_dims: 3 };
        let mut hora = HoraCore::new(config).unwrap();

        // One with embedding, one without
        hora.add_entity("a", "with_emb", None, Some(&[1.0, 0.0, 0.0]))
            .unwrap();
        hora.add_entity("b", "no_emb", None, None).unwrap();

        let results = hora.vector_search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_vector_search_dims_mismatch() {
        let config = HoraConfig { embedding_dims: 3 };
        let hora = HoraCore::new(config).unwrap();

        // Query with wrong dimensions
        let result = hora.vector_search(&[1.0, 0.0], 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_search_dims_zero_errors() {
        let hora = HoraCore::new(HoraConfig::default()).unwrap();
        let result = hora.vector_search(&[1.0, 0.0, 0.0], 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_search_empty_graph() {
        let config = HoraConfig { embedding_dims: 3 };
        let hora = HoraCore::new(config).unwrap();

        let results = hora.vector_search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_vector_search_k_larger_than_corpus() {
        let config = HoraConfig { embedding_dims: 3 };
        let mut hora = HoraCore::new(config).unwrap();

        hora.add_entity("a", "x", None, Some(&[1.0, 0.0, 0.0]))
            .unwrap();

        let results = hora.vector_search(&[1.0, 0.0, 0.0], 100).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_vector_search_scores_descending() {
        let config = HoraConfig { embedding_dims: 3 };
        let mut hora = HoraCore::new(config).unwrap();

        hora.add_entity("a", "x", None, Some(&[1.0, 0.0, 0.0]))
            .unwrap();
        hora.add_entity("b", "y", None, Some(&[0.5, 0.5, 0.0]))
            .unwrap();
        hora.add_entity("c", "z", None, Some(&[0.0, 1.0, 0.0]))
            .unwrap();

        let results = hora.vector_search(&[1.0, 0.0, 0.0], 3).unwrap();
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    // --- v0.2b tests: BM25 Text Search ---

    #[test]
    fn test_text_search_finds_by_name() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        hora.add_entity("project", "hora graph engine", None, None)
            .unwrap();
        hora.add_entity("language", "Rust programming", None, None)
            .unwrap();

        let results = hora.text_search("hora", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_id, EntityId(1));
    }

    #[test]
    fn test_text_search_finds_by_properties() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        hora.add_entity(
            "project",
            "hora",
            Some(props! { "description" => "knowledge graph authentication engine" }),
            None,
        )
        .unwrap();
        hora.add_entity("other", "unrelated", None, None).unwrap();

        let results = hora.text_search("authentication", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_id, EntityId(1));
    }

    #[test]
    fn test_text_search_tf_ranking() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        // Same doc length, but entity 1 has "rust" 3 times vs entity 2's 1 time
        hora.add_entity(
            "a",
            "rust rust rust",
            None,
            None,
        )
        .unwrap();
        hora.add_entity("b", "rust java python", None, None)
            .unwrap();

        let results = hora.text_search("rust", 10).unwrap();
        assert_eq!(results.len(), 2);
        // More occurrences (same length) → higher score
        assert_eq!(results[0].entity_id, EntityId(1));
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_text_search_no_match() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        hora.add_entity("project", "hora", None, None).unwrap();

        let results = hora.text_search("nonexistent", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_text_search_respects_delete() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let id = hora
            .add_entity("project", "hora graph engine", None, None)
            .unwrap();

        hora.delete_entity(id).unwrap();

        let results = hora.text_search("hora", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_text_search_respects_update() {
        let mut hora = HoraCore::new(HoraConfig::default()).unwrap();
        let id = hora
            .add_entity("project", "old name cats", None, None)
            .unwrap();

        hora.update_entity(
            id,
            EntityUpdate {
                name: Some("new name dogs".to_string()),
                ..Default::default()
            },
        )
        .unwrap();

        assert!(hora.text_search("cats", 10).unwrap().is_empty());
        assert_eq!(hora.text_search("dogs", 10).unwrap().len(), 1);
    }

    #[test]
    fn test_text_search_after_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bm25.hora");

        {
            let mut hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            hora.add_entity("project", "hora graph engine", None, None)
                .unwrap();
            hora.add_entity("language", "rust programming", None, None)
                .unwrap();
            hora.flush().unwrap();
        }

        // Reopen → BM25 index rebuilt from entities
        {
            let mut hora = HoraCore::open(&path, HoraConfig::default()).unwrap();
            let results = hora.text_search("hora", 10).unwrap();
            assert_eq!(results.len(), 1);

            let results = hora.text_search("rust", 10).unwrap();
            assert_eq!(results.len(), 1);
        }
    }
}
