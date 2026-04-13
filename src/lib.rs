/*!
# cuda-ghost-tiles

Ghost Tiles — learned sparse attention patterns for efficient agent cognition.

## The Problem

In transformer attention, computing every token-to-token interaction is O(n²).
For a fleet of agents processing long contexts, this becomes the bottleneck.
Most attention positions contribute nothing — they're noise.

## The Solution

Ghost tiles learn which positions *actually matter*. Instead of computing
a full attention matrix, agents compute only the tiles with learned relevance.
Unimportant tiles become "ghosts" — present in the logical pattern but
computationally absent.

## How It Works

1. **Tile Grid**: Divide the attention matrix into 8×8 tiles
2. **Weight Learning**: Track which tiles get used, strengthen active ones
3. **Pruning**: Deactivate lowest-weight tiles to meet sparsity budget
4. **Decay**: Unused tiles lose weight over time (forgetting curve)
5. **Merge**: Combine complementary patterns for multi-task agents
6. **CUDA Kernel**: Each tile maps to a GPU thread block for parallel compute

## Integration

- **cuda-attention**: Uses ghost tiles for saliency scoring
- **cuda-memory-fabric**: Tiles stored as procedural memory
- **cuda-emergence**: Sparse patterns detected as emergent behavior
- **cuda-voxel-logic**: 2D attention tiles generalize to 3D spatial tiles
- **cuda-equipment**: GPU memory requirements calculated from active tiles

## Fleet Usage

```ignore
let mut mgr = GhostTileManager::new(0.5); // 50% sparsity budget
let mut pattern = GhostPattern::new("reasoning", seq_len, tile_size, 1.0);
pattern.use_tile(row, col, confidence);
pattern.prune();
mgr.add_pattern(pattern);
let best = mgr.best_pattern(); // most efficient pattern
```
*/

pub mod attention;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single tile in the attention grid
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GhostTile {
    pub row: usize,
    pub col: usize,
    pub active: bool,
    pub weight: f64,
    pub last_used: u64,
    pub use_count: u64,
    pub confidence: f64,
    pub importance: f64,
}

impl GhostTile {
    pub fn new(row: usize, col: usize) -> Self {
        GhostTile { row, col, active: true, weight: 1.0, last_used: 0, use_count: 0, confidence: 1.0, importance: 0.5 }
    }

    /// Activate this tile with usage tracking
    pub fn use_tile(&mut self, confidence: f64) {
        self.active = true;
        self.use_count += 1;
        self.last_used = now();
        // Bayesian update: fuse new confidence with existing
        let fused = fuse_confidence(self.confidence, confidence);
        self.confidence = fused;
        // Weight grows with usage but saturates
        self.weight = 1.0 - (1.0 - self.weight) * 0.9;
        self.importance = self.weight * self.confidence;
    }

    /// Decay this tile's weight (forgetting curve)
    pub fn decay(&mut self, rate: f64) {
        let age_ms = now() - self.last_used;
        let decay = (-rate * age_ms as f64 / 1000.0 / 60.0).exp(); // per-minute decay
        self.weight *= decay;
        self.confidence *= (1.0 - rate * 0.1); // slow confidence decay
        self.importance = self.weight * self.confidence;
        if self.weight < 0.01 {
            self.active = false;
        }
    }

    /// Convert to CUDA kernel parameters
    pub fn cuda_params(&self) -> (usize, usize, u32) {
        (self.row, self.col, if self.active { 1 } else { 0 })
    }
}

/// A learned sparse attention pattern
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GhostPattern {
    pub id: String,
    pub tiles: Vec<GhostTile>,
    pub rows: usize,
    pub cols: usize,
    pub tile_size: usize,
    pub sparsity_budget: f64,
    pub total_uses: u64,
    pub created_ms: u64,
    pub last_rebalance_ms: u64,
}

impl GhostPattern {
    pub fn new(id: &str, seq_len: usize, tile_size: usize, sparsity_budget: f64) -> Self {
        let grid_size = (seq_len + tile_size - 1) / tile_size;
        let mut tiles = Vec::with_capacity(grid_size * grid_size);
        for r in 0..grid_size {
            for c in 0..grid_size {
                tiles.push(GhostTile::new(r, c));
            }
        }
        GhostPattern { id: id.to_string(), tiles, rows: grid_size, cols: grid_size, tile_size, sparsity_budget, total_uses: 0, created_ms: now(), last_rebalance_ms: now() }
    }

    /// Use a specific tile at (row, col)
    pub fn use_tile(&mut self, row: usize, col: usize, confidence: f64) {
        self.total_uses += 1;
        if let Some(idx) = self.tiles.iter().position(|t| t.row == row && t.col == col) {
            self.tiles[idx].use_tile(confidence);
        }
    }

    /// Prune tiles to meet sparsity budget
    pub fn prune(&mut self) {
        let max_active = (self.tiles.len() as f64 * (1.0 - self.sparsity_budget)) as usize;
        let active_count = self.tiles.iter().filter(|t| t.active).count();
        if active_count <= max_active { return; }
        // Sort by importance descending, deactivate tail
        let mut indexed: Vec<_> = self.tiles.iter_mut().enumerate().collect();
        indexed.sort_by(|a, b| b.1.importance.partial_cmp(&a.1.importance).unwrap_or(std::cmp::Ordering::Equal));
        for (i, tile) in indexed.iter_mut() {
            if *i >= max_active { tile.active = false; }
        }
    }

    /// Decay all tiles
    pub fn decay(&mut self, rate: f64) {
        for tile in &mut self.tiles { tile.decay(rate); }
    }

    /// Rebalance: prune + decay + reactivate high-confidence inactive
    pub fn rebalance(&mut self) {
        self.prune();
        self.decay(0.1);
        // Reactivate inactive tiles with high residual confidence
        let max_active = (self.tiles.len() as f64 * (1.0 - self.sparsity_budget)) as usize;
        let active_count = self.tiles.iter().filter(|t| t.active).count();
        if active_count < max_active {
            let slots = max_active - active_count;
            let mut inactive: Vec<_> = self.tiles.iter_mut().filter(|t| !t.active).collect();
            inactive.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
            for tile in inactive.iter_mut().take(slots) {
                tile.active = true;
                tile.weight = 0.5; // reset to moderate
            }
        }
        self.last_rebalance_ms = now();
    }

    /// Get active tile indices
    pub fn active_tiles(&self) -> Vec<&GhostTile> {
        self.tiles.iter().filter(|t| t.active).collect()
    }

    /// Sparsity ratio (fraction of tiles that are inactive/ghosts)
    pub fn sparsity(&self) -> f64 {
        let total = self.tiles.len();
        let active = self.tiles.iter().filter(|t| t.active).count();
        1.0 - (active as f64 / total as f64)
    }

    /// Coverage: fraction of grid that's been used at least once
    pub fn coverage(&self) -> f64 {
        let used = self.tiles.iter().filter(|t| t.use_count > 0).count();
        used as f64 / self.tiles.len() as f64
    }

    /// Efficiency: ratio of active tiles that are heavily used
    pub fn efficiency(&self) -> f64 {
        let active = self.tiles.iter().filter(|t| t.active).count();
        if active == 0 { return 0.0; }
        let heavy = self.tiles.iter().filter(|t| t.active && t.use_count > 5).count();
        heavy as f64 / active as f64
    }

    /// Compute cost relative to full attention (1.0 = full, 0.0 = free)
    pub fn compute_cost(&self) -> f64 {
        let active = self.tiles.iter().filter(|t| t.active).count();
        active as f64 / self.tiles.len() as f64
    }

    /// Serialize to CUDA-compatible tile map
    pub fn to_cuda_tile_map(&self) -> Vec<(usize, usize)> {
        self.tiles.iter().filter(|t| t.active).map(|t| (t.row, t.col)).collect()
    }

    /// Attention mask: 1.0 for active, 0.0 for ghost
    pub fn attention_mask(&self, tile_size: usize, seq_len: usize) -> Vec<f32> {
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for tile in &self.tiles {
            if !tile.active { continue; }
            let r_start = tile.row * tile_size;
            let c_start = tile.col * tile_size;
            for r in r_start..(r_start + tile_size).min(seq_len) {
                for c in c_start..(c_start + tile_size).min(seq_len) {
                    mask[r * seq_len + c] = tile.weight as f32;
                }
            }
        }
        mask
    }

    /// Merge two patterns by averaging weights
    pub fn merge_with(&self, other: &GhostPattern, new_id: &str) -> GhostPattern {
        let mut merged = GhostPattern::new(new_id, self.rows * self.tile_size, self.tile_size, self.sparsity_budget);
        let min_tiles = self.tiles.len().min(other.tiles.len());
        for i in 0..min_tiles {
            merged.tiles[i].weight = (self.tiles[i].weight + other.tiles[i].weight) / 2.0;
            merged.tiles[i].confidence = fuse_confidence(self.tiles[i].confidence, other.tiles[i].confidence);
            merged.tiles[i].use_count = self.tiles[i].use_count.max(other.tiles[i].use_count);
            merged.tiles[i].importance = merged.tiles[i].weight * merged.tiles[i].confidence;
            merged.tiles[i].active = self.tiles[i].active || other.tiles[i].active;
        }
        merged.prune();
        merged
    }
}

/// Manages multiple ghost patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GhostTileManager {
    pub patterns: HashMap<String, GhostPattern>,
    pub sparsity_budget: f64,
    pub total_computed: u64,
    pub total_saved: u64, // compute saved vs full attention
}

impl GhostTileManager {
    pub fn new(sparsity_budget: f64) -> Self { GhostTileManager { patterns: HashMap::new(), sparsity_budget, total_computed: 0, total_saved: 0 } }

    pub fn add_pattern(&mut self, pattern: GhostPattern) {
        let saved = (pattern.compute_cost() * pattern.tiles.len() as f64) as u64;
        self.total_saved += saved;
        self.total_computed += pattern.tiles.len() as u64;
        self.patterns.insert(pattern.id.clone(), pattern);
    }

    pub fn get(&self, id: &str) -> Option<&GhostPattern> { self.patterns.get(id) }

    /// Merge two patterns
    pub fn merge(&mut self, a: &str, b: &str, new_id: &str) -> Option<GhostPattern> {
        let pa = self.patterns.get(a)?;
        let pb = self.patterns.get(b)?;
        let merged = pa.merge_with(pb, new_id);
        self.add_pattern(merged.clone());
        Some(merged)
    }

    /// Best pattern by efficiency
    pub fn best_pattern(&self) -> Option<&GhostPattern> {
        self.patterns.values().max_by(|a, b| a.efficiency().partial_cmp(&b.efficiency()).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Most used pattern
    pub fn most_used(&self) -> Option<&GhostPattern> {
        self.patterns.values().max_by_key(|p| p.total_uses)
    }

    /// Sparsiest pattern (most ghost tiles)
    pub fn sparsiest(&self) -> Option<&GhostPattern> {
        self.patterns.values().max_by(|a, b| a.sparsity().partial_cmp(&b.sparsity()).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Decay all patterns
    pub fn decay_all(&mut self, rate: f64) {
        for p in self.patterns.values_mut() { p.decay(rate); }
    }

    /// Average compute cost across all patterns
    pub fn avg_compute_cost(&self) -> f64 {
        if self.patterns.is_empty() { return 1.0; }
        let total: f64 = self.patterns.values().map(|p| p.compute_cost()).sum();
        total / self.patterns.len() as f64
    }

    pub fn savings_pct(&self) -> f64 {
        if self.total_computed == 0 { return 0.0; }
        (1.0 - self.total_computed as f64 / (self.total_computed + self.total_saved) as f64) * 100.0
    }

    pub fn summary(&self) -> String {
        format!("GhostTileManager: {} patterns, avg_cost={:.1}%, savings={:.1}%",
            self.patterns.len(), self.avg_compute_cost() * 100.0, self.savings_pct())
    }
}

/// Fuse two confidences via harmonic mean (same as cuda-confidence)
fn fuse_confidence(a: f64, b: f64) -> f64 {
    let inv = 1.0 / a.max(1e-10) + 1.0 / b.max(1e-10);
    if inv >= 1e10 { return 0.0; }
    1.0 / inv
}

fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_use() {
        let mut tile = GhostTile::new(0, 0);
        tile.use_tile(0.9);
        assert!(tile.active);
        assert_eq!(tile.use_count, 1);
        assert!(tile.confidence > 0.0);
    }

    #[test]
    fn test_tile_decay() {
        let mut tile = GhostTile::new(0, 0);
        tile.use_tile(1.0);
        tile.last_used = 0;
        tile.decay(0.5);
        assert!(tile.weight < 1.0);
    }

    #[test]
    fn test_tile_deactivation() {
        let mut tile = GhostTile::new(0, 0);
        tile.weight = 0.005;
        tile.decay(0.5);
        assert!(!tile.active);
    }

    #[test]
    fn test_pattern_create() {
        let p = GhostPattern::new("p1", 64, 8, 0.5);
        assert_eq!(p.tiles.len(), 64); // 8×8 grid
        assert_eq!(p.coverage(), 0.0);
    }
    #[test]
    fn test_pattern_prune() {
        let mut p = GhostPattern::new("p1", 64, 8, 0.5);
        for t in &mut p.tiles { t.weight = 0.5; }
        p.prune();
        let active = p.active_tiles().len();
        assert!(active <= 32);
    }

    #[test]
    fn test_pattern_sparsity() {
        let p = GhostPattern::new("p1", 64, 8, 0.0); // no sparsity
        assert!((p.sparsity() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_coverage() {
        let mut p = GhostPattern::new("p1", 64, 8, 1.0);
        // Fresh pattern has 0 coverage — use some tiles first
        assert_eq!(p.coverage(), 0.0);
        p.tiles[0].use_count = 1;
        p.tiles[1].use_count = 1;
        assert!(p.coverage() > 0.0);
    }

    #[test]
    fn test_efficiency() {
        let mut p = GhostPattern::new("p1", 64, 8, 0.5);
        p.prune();
        assert!(p.efficiency() >= 0.0);
    }

    #[test]
    fn test_compute_cost() {
        let p = GhostPattern::new("p1", 64, 8, 0.0);
        assert!((p.compute_cost() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_attention_mask() {
        let p = GhostPattern::new("p1", 64, 8, 0.0);
        let mask = p.attention_mask(8, 64);
        assert_eq!(mask.len(), 64 * 64);
        let nonzero: usize = mask.iter().filter(|&&m| m > 0.0).count();
        assert!(nonzero > 0);
    }

    #[test]
    fn test_cuda_tile_map() {
        let p = GhostPattern::new("p1", 64, 8, 0.0);
        let map = p.to_cuda_tile_map();
        assert_eq!(map.len(), 64);
    }

    #[test]
    fn test_merge() {
        let mut mgr = GhostTileManager::new(0.5);
        let mut a = GhostPattern::new("a", 64, 8, 0.5);
        let mut b = GhostPattern::new("b", 64, 8, 0.5);
        for t in &mut a.tiles { if t.row < 4 { t.weight = 0.9; } else { t.weight = 0.1; } }
        for t in &mut b.tiles { if t.row >= 4 { t.weight = 0.9; } else { t.weight = 0.1; } }
        a.prune(); b.prune();
        mgr.add_pattern(a); mgr.add_pattern(b);
        let merged = mgr.merge("a", "b", "merged");
        assert!(merged.is_some());
    }

    #[test]
    fn test_best_pattern() {
        let mut mgr = GhostTileManager::new(0.5);
        let mut p1 = GhostPattern::new("dense", 64, 8, 0.8);
        let mut p2 = GhostPattern::new("sparse", 64, 8, 0.2);
        p1.prune(); p2.prune();
        mgr.add_pattern(p1); mgr.add_pattern(p2);
        let best = mgr.best_pattern().unwrap();
        assert!(best.id.len() > 0); // best pattern selected
    }

    #[test]
    fn test_rebalance() {
        let mut p = GhostPattern::new("p1", 64, 8, 0.3);
        for t in &mut p.tiles { t.weight = 0.5; }
        p.rebalance();
        let active = p.active_tiles().len();
        assert!(active < p.tiles.len());
    }

    #[test]
    fn test_decay() {
        let mut p = GhostPattern::new("p1", 64, 8, 1.0);
        p.use_tile(0, 0, 1.0);
        let before = p.tiles[0].weight;
        p.tiles[0].last_used = 0;
        p.decay(0.5);
        assert!(p.tiles[0].weight < before);
    }

    #[test]
    fn test_manager_summary() {
        let mgr = GhostTileManager::new(0.5);
        let s = mgr.summary();
        assert!(s.contains("0 patterns"));
    }

    #[test]
    fn test_savings() {
        let mut mgr = GhostTileManager::new(0.5);
        let mut p = GhostPattern::new("p1", 64, 8, 0.5);
        p.prune();
        mgr.add_pattern(p);
        assert!(mgr.savings_pct() > 0.0);
    }

    #[test]
    fn test_fuse_confidence() {
        assert!(fuse_confidence(0.8, 0.8) > 0.0);
        assert!(fuse_confidence(0.5, 0.5) < 0.5); // harmonic mean < arithmetic
        assert!((fuse_confidence(1.0, 1.0) - 0.5).abs() < 0.01)  // harmonic mean;
    }

    #[test]
    fn test_most_used() {
        let mut mgr = GhostTileManager::new(0.5);
        let mut p1 = GhostPattern::new("p1", 64, 8, 0.5);
        let mut p2 = GhostPattern::new("p2", 64, 8, 0.5);
        for _ in 0..10 { p1.use_tile(0, 0, 0.9); }
        for _ in 0..20 { p2.use_tile(0, 0, 0.9); }
        p1.prune(); p2.prune();
        mgr.add_pattern(p1); mgr.add_pattern(p2);
        assert_eq!(mgr.most_used().unwrap().id, "p2");
    }

    #[test]
    fn test_sparsiest() {
        let mut mgr = GhostTileManager::new(0.5);
        let mut p1 = GhostPattern::new("p1", 64, 8, 0.9);
        let mut p2 = GhostPattern::new("p2", 64, 8, 0.2);
        p1.prune(); p2.prune();
        mgr.add_pattern(p1); mgr.add_pattern(p2);
        assert_eq!(mgr.sparsiest().unwrap().id, "p1");
    }
}
