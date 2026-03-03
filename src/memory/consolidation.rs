//! Consolidation parameters for the dream cycle.
//!
//! SHY (Synaptic Homeostasis Hypothesis) — Tononi & Cirelli 2003, 2006.
//! During sleep, synaptic weights are globally downscaled to restore
//! signal-to-noise ratio while preserving relative differences.

/// Default SHY downscaling factor (≈22% reduction per cycle).
pub const DEFAULT_SHY_FACTOR: f64 = 0.78;

/// Parameters for the SHY downscaling pass.
#[derive(Debug, Clone)]
pub struct ConsolidationParams {
    /// Multiplicative factor applied to all activation scores (default 0.78).
    pub shy_factor: f64,
}

impl Default for ConsolidationParams {
    fn default() -> Self {
        Self {
            shy_factor: DEFAULT_SHY_FACTOR,
        }
    }
}
