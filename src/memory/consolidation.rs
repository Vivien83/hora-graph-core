//! Consolidation parameters for the dream cycle.
//!
//! SHY (Synaptic Homeostasis Hypothesis) — Tononi & Cirelli 2003, 2006.
//! During sleep, synaptic weights are globally downscaled to restore
//! signal-to-noise ratio while preserving relative differences.
//!
//! Interleaved Replay — McClelland et al. 1995, Ji & Wilson 2007.
//! Hippocampal replay interleaves recent and old episodes to prevent
//! catastrophic interference during consolidation.
//!
//! CLS Transfer — McClelland et al. 1995, Kumaran et al. 2016.
//! Complementary Learning Systems: recurring episodic patterns are
//! extracted into semantic facts (neocortical consolidation).

/// Default SHY downscaling factor (≈22% reduction per cycle).
pub const DEFAULT_SHY_FACTOR: f64 = 0.78;

/// Default ratio of recent episodes in interleaved replay.
pub const DEFAULT_RECENT_RATIO: f64 = 0.7;

/// Default maximum number of episodes replayed per cycle.
pub const DEFAULT_MAX_REPLAY: usize = 100;

/// Default CLS transfer threshold (minimum consolidation_count to consider).
pub const DEFAULT_CLS_THRESHOLD: u32 = 3;

/// Parameters for the consolidation cycle.
#[derive(Debug, Clone)]
pub struct ConsolidationParams {
    /// Multiplicative factor applied to all activation scores (default 0.78).
    pub shy_factor: f64,
    /// Ratio of recent episodes in interleaved replay (default 0.7 = 70%).
    pub recent_ratio: f64,
    /// Maximum number of episodes replayed per cycle (default 100).
    pub max_replay_items: usize,
    /// Minimum consolidation_count for CLS transfer eligibility (default 3).
    pub cls_threshold: u32,
}

impl Default for ConsolidationParams {
    fn default() -> Self {
        Self {
            shy_factor: DEFAULT_SHY_FACTOR,
            recent_ratio: DEFAULT_RECENT_RATIO,
            max_replay_items: DEFAULT_MAX_REPLAY,
            cls_threshold: DEFAULT_CLS_THRESHOLD,
        }
    }
}

/// Statistics returned by interleaved_replay().
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayStats {
    /// Number of episodes replayed.
    pub episodes_replayed: usize,
    /// Number of entity reactivations triggered (one per entity per episode).
    pub entities_reactivated: usize,
}

/// Statistics returned by cls_transfer().
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClsStats {
    /// Number of episodes processed (consolidation_count >= threshold).
    pub episodes_processed: usize,
    /// Number of new semantic facts created.
    pub facts_created: usize,
    /// Number of existing semantic facts reinforced (confidence bumped).
    pub facts_reinforced: usize,
}
