/*!
Task routing engine using attention scores.

The router takes scored agents and makes final routing decisions,
respecting constraints like load balancing, task affinity, and
capacity limits.
*/

use super::agent_profile::{AgentId, AgentProfile};
use super::scorer::{AttentionScore, AttentionScorer, ScorerConfig};
use super::task_embedding::Task;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The result of routing a task to an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// The task being routed
    pub task_id: String,
    /// The selected agent
    pub agent_id: AgentId,
    /// Attention score for this decision
    pub score: AttentionScore,
    /// Confidence in the routing decision (0.0-1.0)
    /// Based on how dominant the best score is vs. alternatives
    pub confidence: f64,
    /// Routing strategy that produced this decision
    pub strategy: RoutingStrategy,
    /// Why this agent was selected
    pub reason: String,
}

/// Available routing strategies
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route to the single best-matching agent
    BestMatch,
    /// Route to the best-matching available (idle) agent
    BestAvailable,
    /// Route using round-robin among qualified agents
    RoundRobin,
    /// Route to distribute load evenly across agents
    LoadBalanced,
    /// Route using softmax-weighted probabilistic selection
    Probabilistic,
}

impl std::fmt::Display for RoutingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoutingStrategy::BestMatch => write!(f, "best_match"),
            RoutingStrategy::BestAvailable => write!(f, "best_available"),
            RoutingStrategy::RoundRobin => write!(f, "round_robin"),
            RoutingStrategy::LoadBalanced => write!(f, "load_balanced"),
            RoutingStrategy::Probabilistic => write!(f, "probabilistic"),
        }
    }
}

/// Routing statistics for monitoring and diagnostics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RoutingStats {
    /// Total tasks routed
    pub total_routed: u64,
    /// Tasks routed per agent
    pub routes_per_agent: HashMap<AgentId, u64>,
    /// Average confidence across all routing decisions
    pub avg_confidence: f64,
    /// Total confidence (running sum for computing avg)
    pub total_confidence: f64,
    /// Number of failed routing attempts (no agent available)
    pub failed_routes: u64,
    /// Strategy usage counts
    pub strategy_usage: HashMap<String, u64>,
}

impl RoutingStats {
    pub fn record_route(&mut self, decision: &RoutingDecision) {
        self.total_routed += 1;
        self.total_confidence += decision.confidence;
        self.avg_confidence = self.total_confidence / self.total_routed as f64;
        *self
            .routes_per_agent
            .entry(decision.agent_id.clone())
            .or_insert(0) += 1;
        *self
            .strategy_usage
            .entry(decision.strategy.to_string())
            .or_insert(0) += 1;
    }

    pub fn record_failure(&mut self) {
        self.failed_routes += 1;
    }

    /// Success rate: tasks successfully routed / total attempts
    pub fn success_rate(&self) -> f64 {
        let total = self.total_routed + self.failed_routes;
        if total == 0 {
            return 0.0;
        }
        self.total_routed as f64 / total as f64
    }
}

/// Task routing engine
pub struct TaskRouter {
    scorer: AttentionScorer,
    strategy: RoutingStrategy,
    stats: RoutingStats,
    /// Round-robin counter
    rr_counter: usize,
    /// Random seed for deterministic probabilistic routing
    seed: u64,
}

impl TaskRouter {
    /// Create a new task router
    pub fn new(scorer: AttentionScorer) -> Self {
        TaskRouter {
            scorer,
            strategy: RoutingStrategy::BestAvailable,
            stats: RoutingStats::default(),
            rr_counter: 0,
            seed: 42,
        }
    }

    /// Create a router with default configuration
    pub fn with_defaults() -> Self {
        TaskRouter::new(AttentionScorer::new(ScorerConfig::default()))
    }

    /// Set the routing strategy
    pub fn with_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Get the current routing statistics
    pub fn stats(&self) -> &RoutingStats {
        &self.stats
    }

    /// Get mutable reference to the scorer for configuration
    pub fn scorer_mut(&mut self) -> &mut AttentionScorer {
        &mut self.scorer
    }

    /// Route a single task to the best available agent.
    ///
    /// Returns None if no suitable agent is available.
    pub fn route(&mut self, task: &Task, agents: &[AgentProfile]) -> Option<RoutingDecision> {
        // First, filter to available agents
        let available: Vec<&AgentProfile> = agents.iter().filter(|a| a.is_available()).collect();
        if available.is_empty() {
            self.stats.record_failure();
            return None;
        }

        let available_refs: Vec<AgentProfile> = available.into_iter().cloned().collect();

        match self.strategy {
            RoutingStrategy::BestMatch => self.route_best_match(task, agents),
            RoutingStrategy::BestAvailable => self.route_best_match(task, &available_refs),
            RoutingStrategy::RoundRobin => self.route_round_robin(task, &available_refs),
            RoutingStrategy::LoadBalanced => self.route_load_balanced(task, &available_refs),
            RoutingStrategy::Probabilistic => self.route_probabilistic(task, &available_refs),
        }
    }

    /// Route a batch of tasks, respecting capacity constraints.
    /// Each task is routed to a different agent when possible.
    pub fn route_batch(
        &mut self,
        tasks: &[Task],
        agents: &[AgentProfile],
    ) -> Vec<Option<RoutingDecision>> {
        // Track remaining capacity per agent
        let mut remaining: HashMap<AgentId, u32> = agents
            .iter()
            .map(|a| (a.id.clone(), a.available_slots()))
            .collect();

        let mut decisions = Vec::with_capacity(tasks.len());

        for task in tasks {
            // Filter agents with remaining capacity
            let eligible: Vec<AgentProfile> = agents
                .iter()
                .filter(|a| a.is_available() && *remaining.get(&a.id).unwrap_or(&0) > 0)
                .cloned()
                .collect();

            if let Some(mut decision) = self.route(task, &eligible) {
                // Decrement remaining capacity
                if let Some(slots) = remaining.get_mut(&decision.agent_id) {
                    *slots = slots.saturating_sub(1);
                }
                decision.reason = format!(
                    "{} (batch routing, {} slots remaining)",
                    decision.reason,
                    remaining.get(&decision.agent_id).unwrap_or(&0)
                );
                decisions.push(Some(decision));
            } else {
                decisions.push(None);
            }
        }

        decisions
    }

    fn route_best_match(
        &mut self,
        task: &Task,
        agents: &[AgentProfile],
    ) -> Option<RoutingDecision> {
        let scores = self.scorer.score(task, agents);
        if scores.is_empty() {
            self.stats.record_failure();
            return None;
        }

        let best = scores
            .into_iter()
            .max_by(|a, b| {
                a.adjusted_score
                    .partial_cmp(&b.adjusted_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;

        let confidence = compute_confidence(&best, 1.0);

        let decision = RoutingDecision {
            task_id: task.id.clone(),
            agent_id: best.agent_id.clone(),
            score: best,
            confidence,
            strategy: RoutingStrategy::BestMatch,
            reason: "Highest attention score".to_string(),
        };

        self.stats.record_route(&decision);
        Some(decision)
    }

    fn route_round_robin(
        &mut self,
        task: &Task,
        agents: &[AgentProfile],
    ) -> Option<RoutingDecision> {
        let scores = self.scorer.score(task, agents);
        if scores.is_empty() {
            self.stats.record_failure();
            return None;
        }

        // Sort by score descending for round-robin among qualified
        let mut sorted: Vec<_> = scores;
        sorted.sort_by(|a, b| {
            b.adjusted_score
                .partial_cmp(&a.adjusted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Round-robin selection
        let idx = self.rr_counter % sorted.len();
        self.rr_counter += 1;

        let selected = sorted.remove(idx);
        let confidence = compute_confidence(&selected, sorted.len() as f64);

        let decision = RoutingDecision {
            task_id: task.id.clone(),
            agent_id: selected.agent_id.clone(),
            score: selected,
            confidence,
            strategy: RoutingStrategy::RoundRobin,
            reason: format!("Round-robin selection (index {})", idx),
        };

        self.stats.record_route(&decision);
        Some(decision)
    }

    fn route_load_balanced(
        &mut self,
        task: &Task,
        agents: &[AgentProfile],
    ) -> Option<RoutingDecision> {
        let scores = self.scorer.score(task, agents);
        if scores.is_empty() {
            self.stats.record_failure();
            return None;
        }

        // Sort by (available_slots descending, then adjusted_score descending)
        let mut sorted: Vec<_> = scores;
        sorted.sort_by(|a, b| {
            // Prefer agents with more available slots
            // Since we only have load info in the breakdown, use load_penalty as proxy
            let load_a = a.breakdown.load_penalty;
            let load_b = b.breakdown.load_penalty;
            // Lower load first, then higher score
            match load_a.partial_cmp(&load_b) {
                Some(std::cmp::Ordering::Equal) | None => {
                    b.adjusted_score
                        .partial_cmp(&a.adjusted_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                Some(ord) => ord,
            }
        });

        let best = sorted.into_iter().next()?;
        let confidence = compute_confidence(&best, 1.0);

        let decision = RoutingDecision {
            task_id: task.id.clone(),
            agent_id: best.agent_id.clone(),
            score: best,
            confidence,
            strategy: RoutingStrategy::LoadBalanced,
            reason: "Lowest load among qualified agents".to_string(),
        };

        self.stats.record_route(&decision);
        Some(decision)
    }

    fn route_probabilistic(
        &mut self,
        task: &Task,
        agents: &[AgentProfile],
    ) -> Option<RoutingDecision> {
        let scores = self.scorer.score(task, agents);
        if scores.is_empty() {
            self.stats.record_failure();
            return None;
        }

        // Weighted random selection based on attention weights
        let weights: Vec<f64> = scores.iter().map(|s| s.attention_weight).collect();
        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            self.stats.record_failure();
            return None;
        }

        // Simple deterministic "random" using seed
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = (self.seed >> 33) as f64 / (1u64 << 31) as f64;
        let mut cumulative = 0.0;
        let mut selected_idx = 0;
        for (i, w) in weights.iter().enumerate() {
            cumulative += w / total_weight;
            if r < cumulative {
                selected_idx = i;
                break;
            }
        }

        let scores_len = scores.len() as f64;
        let selected = scores.into_iter().nth(selected_idx)?;
        let weight = selected.attention_weight;
        let confidence = compute_confidence(&selected, scores_len);

        let decision = RoutingDecision {
            task_id: task.id.clone(),
            agent_id: selected.agent_id.clone(),
            score: selected,
            confidence,
            strategy: RoutingStrategy::Probabilistic,
            reason: format!(
                "Probabilistic selection (weight={:.3})",
                weight
            ),
        };

        self.stats.record_route(&decision);
        Some(decision)
    }
}

/// Compute routing confidence: how dominant is the best score vs. alternatives?
/// confidence = 1 - (entropy / max_entropy)
/// For a single option, confidence = 1.0
fn compute_confidence(best: &AttentionScore, num_alternatives: f64) -> f64 {
    if num_alternatives <= 1.0 {
        return 1.0;
    }
    // Simple confidence: the attention weight of the best score
    // Higher weight = more confident
    best.attention_weight.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::agent_profile::{AgentProfileBuilder, SkillTag};
    use crate::attention::task_embedding::TaskPriority;

    fn make_agents() -> Vec<AgentProfile> {
        vec![
            AgentProfileBuilder::new("coder", "CodeBot")
                .max_capacity(3)
                .skill(SkillTag::Coding, 0.95)
                .skill(SkillTag::Testing, 0.7)
                .build(),
            AgentProfileBuilder::new("vision", "VisBot")
                .max_capacity(2)
                .skill(SkillTag::Vision, 0.9)
                .skill(SkillTag::Creative, 0.8)
                .build(),
            AgentProfileBuilder::new("math", "MathBot")
                .max_capacity(2)
                .skill(SkillTag::Mathematics, 0.95)
                .skill(SkillTag::DataAnalysis, 0.8)
                .build(),
        ]
    }

    fn coding_task(id: &str) -> Task {
        Task::new(id, "Write code", TaskPriority::Normal, 0.5)
            .require_skill(SkillTag::Coding, 0.8)
    }

    #[test]
    fn test_route_basic() {
        let mut router = TaskRouter::with_defaults();
        let agents = make_agents();
        let task = coding_task("t1");

        let decision = router.route(&task, &agents);
        assert!(decision.is_some());
        let d = decision.unwrap();
        assert_eq!(d.task_id, "t1");
        assert_eq!(d.agent_id, "coder");
        assert!(d.confidence > 0.0);
    }

    #[test]
    fn test_route_no_agents() {
        let mut router = TaskRouter::with_defaults();
        let task = coding_task("t1");
        let decision = router.route(&task, &[]);
        assert!(decision.is_none());
        assert_eq!(router.stats().failed_routes, 1);
    }

    #[test]
    fn test_route_all_offline() {
        let mut router = TaskRouter::with_defaults();
        let mut agents = make_agents();
        for a in &mut agents {
            a.online = false;
        }
        let task = coding_task("t1");
        let decision = router.route(&task, &agents);
        assert!(decision.is_none());
    }

    #[test]
    fn test_route_all_full() {
        let mut router = TaskRouter::with_defaults();
        let mut agents = make_agents();
        for a in &mut agents {
            a.set_load(1.0);
        }
        let task = coding_task("t1");
        let decision = router.route(&task, &agents);
        assert!(decision.is_none());
    }

    #[test]
    fn test_route_best_match_strategy() {
        let mut router = TaskRouter::with_defaults().with_strategy(RoutingStrategy::BestMatch);
        let agents = make_agents();
        let task = coding_task("t1");

        let decision = router.route(&task, &agents).unwrap();
        assert_eq!(decision.strategy, RoutingStrategy::BestMatch);
        assert_eq!(decision.agent_id, "coder");
    }

    #[test]
    fn test_route_round_robin_strategy() {
        let mut router = TaskRouter::with_defaults().with_strategy(RoutingStrategy::RoundRobin);

        // Use agents with overlapping skills so multiple qualify
        let agents = vec![
            AgentProfileBuilder::new("coder", "CodeBot")
                .max_capacity(3)
                .skill(SkillTag::Coding, 0.95)
                .skill(SkillTag::Testing, 0.7)
                .build(),
            AgentProfileBuilder::new("coder2", "CodeBot2")
                .max_capacity(3)
                .skill(SkillTag::Coding, 0.85)
                .skill(SkillTag::Testing, 0.6)
                .build(),
            AgentProfileBuilder::new("coder3", "CodeBot3")
                .max_capacity(3)
                .skill(SkillTag::Coding, 0.75)
                .skill(SkillTag::Testing, 0.5)
                .build(),
        ];

        // Route multiple tasks — should cycle through agents
        let mut assigned = std::collections::HashSet::new();
        for i in 0..5 {
            let task = coding_task(&format!("t{}", i));
            if let Some(d) = router.route(&task, &agents) {
                assigned.insert(d.agent_id);
            }
        }
        // Round-robin should distribute across agents
        assert!(assigned.len() > 1);
    }

    #[test]
    fn test_route_load_balanced_strategy() {
        let mut router = TaskRouter::with_defaults().with_strategy(RoutingStrategy::LoadBalanced);

        // Use agents with overlapping skills so load balancing is visible
        let mut agents = vec![
            AgentProfileBuilder::new("coder", "CodeBot")
                .max_capacity(3)
                .skill(SkillTag::Coding, 0.95)
                .build(),
            AgentProfileBuilder::new("coder2", "CodeBot2")
                .max_capacity(3)
                .skill(SkillTag::Coding, 0.85)
                .build(),
        ];
        agents[0].set_load(0.9); // coder is busy
        agents[1].set_load(0.0); // coder2 is idle

        let task = coding_task("t1");
        let decision = router.route(&task, &agents);
        assert!(decision.is_some());
        let d = decision.unwrap();
        assert_eq!(d.strategy, RoutingStrategy::LoadBalanced);
        // The idle agent should be preferred for load balancing
        assert_eq!(d.agent_id, "coder2");
    }

    #[test]
    fn test_route_probabilistic_strategy() {
        let mut router =
            TaskRouter::with_defaults().with_strategy(RoutingStrategy::Probabilistic);
        let agents = make_agents();
        let task = coding_task("t1");

        let decision = router.route(&task, &agents);
        assert!(decision.is_some());
        let d = decision.unwrap();
        assert_eq!(d.strategy, RoutingStrategy::Probabilistic);
    }

    #[test]
    fn test_route_batch() {
        let mut router = TaskRouter::with_defaults();
        let agents = make_agents();

        let tasks = vec![
            coding_task("t1"),
            coding_task("t2"),
            coding_task("t3"),
            coding_task("t4"),
        ];

        let decisions = router.route_batch(&tasks, &agents);

        assert_eq!(decisions.len(), 4);
        // At least some should succeed
        let successes: Vec<_> = decisions.into_iter().filter_map(|d| d).collect();
        assert!(!successes.is_empty());
    }

    #[test]
    fn test_routing_stats() {
        let mut router = TaskRouter::with_defaults();
        let agents = make_agents();

        // Route some tasks
        for i in 0..3 {
            let task = coding_task(&format!("t{}", i));
            let _ = router.route(&task, &agents);
        }

        let stats = router.stats();
        assert_eq!(stats.total_routed, 3);
        assert!(stats.avg_confidence > 0.0);
        assert!(stats.success_rate() > 0.0);
    }

    #[test]
    fn test_routing_stats_failure() {
        let mut router = TaskRouter::with_defaults();
        let task = coding_task("t1");

        // Route with no agents → failure
        let _ = router.route(&task, &[]);

        let stats = router.stats();
        assert_eq!(stats.failed_routes, 1);
        assert!((stats.success_rate() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(format!("{}", RoutingStrategy::BestMatch), "best_match");
        assert_eq!(format!("{}", RoutingStrategy::LoadBalanced), "load_balanced");
    }

    #[test]
    fn test_routing_decision_serialization() {
        let d = RoutingDecision {
            task_id: "t1".to_string(),
            agent_id: "a1".to_string(),
            score: AttentionScore {
                agent_id: "a1".to_string(),
                raw_score: 0.8,
                adjusted_score: 0.75,
                attention_weight: 0.7,
                breakdown: crate::attention::scorer::ScoreBreakdown {
                    skill_match: 0.9,
                    load_penalty: 0.1,
                    experience_bonus: 0.05,
                    skill_gap_penalty: 0.0,
                },
            },
            confidence: 0.7,
            strategy: RoutingStrategy::BestMatch,
            reason: "test".to_string(),
        };

        let json = serde_json::to_string(&d).unwrap();
        let parsed: RoutingDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.task_id, "t1");
        assert_eq!(parsed.agent_id, "a1");
    }
}
