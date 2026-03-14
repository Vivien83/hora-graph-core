//! Search subsystem — vector (SIMD cosine), BM25+, and hybrid RRF fusion.

pub mod bm25;
pub mod hybrid;
pub mod vector;

use crate::core::types::EntityId;

/// A search result: entity ID + similarity score.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct SearchHit {
    /// ID of the matching entity.
    pub entity_id: EntityId,
    /// Relevance score (higher is better; scale depends on the search method).
    pub score: f32,
}

/// Options for hybrid search.
#[derive(Debug, Clone)]
pub struct SearchOpts {
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Include dark nodes in results (for future ACT-R integration).
    pub include_dark: bool,
}

impl Default for SearchOpts {
    fn default() -> Self {
        Self {
            top_k: 10,
            include_dark: false,
        }
    }
}

impl SearchOpts {
    /// Create search options with a result limit.
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            ..Default::default()
        }
    }

    /// Include dark (low-activation) nodes in results.
    pub fn with_dark(mut self) -> Self {
        self.include_dark = true;
        self
    }
}
