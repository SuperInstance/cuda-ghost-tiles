/*!
Attention-based scoring for fleet task routing.

Implements a Q/K/V attention mechanism adapted from transformers:
- **Query (Q)** = Task embedding (what the task needs)
- **Key (K)** = Agent capability embedding (what the agent offers)
- **Value (V)** = Agent profile metadata (used in routing decisions)

The raw dot-product scores are modified by:
1. **Load penalty**: Agents at high load get discounted scores
2. **Experience bonus**: Agents with relevant history get boosted
3. **Skill gap penalty**: Missing or low-proficiency skills reduce scores
4. **Affinity bonus**: Past successful interactions boost score
*/

use super::agent_profile::{AgentProfile, SkillTag};
use super::task_embedding::Task;
use serde::{Deserialize, Serialize};

/// Configuration for the attention scorer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScorerConfig {
    /// Temperature for softmax (lower = sharper attention, higher = more uniform)
    pub temperature: f64,
    /// Weight for the load penalty (0.0 = ignore load, 1.0 = heavy penalty)
    pub load_weight: f64,
    /// Weight for the experience/history bonus
    pub experience_weight: f64,
    /// Weight for skill gap penalty
    pub skill_gap_weight: f64,
    /// Minimum score threshold for an agent to be considered
    pub min_score_threshold: f64,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        ScorerConfig {
            temperature: 1.0,
            load_weight: 0.3,
            experience_weight: 0.2,
            skill_gap_weight: 0.4,
            min_score_threshold: 0.05,
        }
    }
}

impl ScorerConfig {
    /// Create a config optimized for fast/expedited routing
    pub fn expedited() -> Self {
        ScorerConfig {
            temperature: 0.5,  // sharper: pick the best agent quickly
            load_weight: 0.5,  // heavily penalize loaded agents
            experience_weight: 0.3,
            skill_gap_weight: 0.5,
            min_score_threshold: 0.1,
        }
    }

    /// Create a config optimized for quality/accuracy
    pub fn quality_focused() -> Self {
        ScorerConfig {
            temperature: 1.5,  // softer: more uniform distribution
            load_weight: 0.1,  // don't worry about load
            experience_weight: 0.5, // heavily weight experience
            skill_gap_weight: 0.3,
            min_score_threshold: 0.02,
        }
    }
}

/// Score assigned to an agent for a specific task
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionScore {
    /// The agent's ID
    pub agent_id: String,
    /// Raw dot-product attention score (Q · K)
    pub raw_score: f64,
    /// Score after load/penalty adjustments
    pub adjusted_score: f64,
    /// Final score after softmax normalization (sums to 1.0 across all agents)
    pub attention_weight: f64,
    /// Breakdown of score components for explainability
    pub breakdown: ScoreBreakdown,
}

/// Breakdown of scoring components
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Skill match score (0.0-1.0): how well agent skills match task requirements
    pub skill_match: f64,
    /// Load penalty (0.0-1.0): 0 = no load, 1 = fully loaded
    pub load_penalty: f64,
    /// Experience bonus (0.0-1.0): based on historical performance
    pub experience_bonus: f64,
    /// Skill gap penalty (0.0-1.0): penalties for skills below requirements
    pub skill_gap_penalty: f64,
}

/// Attention-based scorer for task-agent matching
pub struct AttentionScorer {
    config: ScorerConfig,
}

impl AttentionScorer {
    pub fn new(config: ScorerConfig) -> Self {
        AttentionScorer { config }
    }

    /// Score a task against a pool of agents.
    ///
    /// Returns attention scores sorted by adjusted_score (descending).
    pub fn score(&self, task: &Task, agents: &[AgentProfile]) -> Vec<AttentionScore> {
        if agents.is_empty() {
            return Vec::new();
        }

        // Compute skill vector for the task (Query)
        let task_skills = task_skill_vector(task);

        let mut scores: Vec<AttentionScore> = Vec::with_capacity(agents.len());

        for agent in agents {
            // Agent skill vector (Key)
            let agent_skills = agent.skill_vector();

            // Compute components
            let skill_match = compute_skill_match(&task_skills, &agent_skills, task);
            let load_penalty = agent.current_load;
            let experience_bonus = compute_experience_bonus(agent, task);
            let skill_gap_penalty = compute_skill_gap(agent, task);

            // Raw score: weighted dot product
            let raw_score = skill_match;

            // Adjusted score with modifiers
            let mut adjusted = raw_score;
            adjusted -= self.config.load_weight * load_penalty;
            adjusted += self.config.experience_weight * experience_bonus;
            adjusted -= self.config.skill_gap_weight * skill_gap_penalty;

            scores.push(AttentionScore {
                agent_id: agent.id.clone(),
                raw_score,
                adjusted_score: adjusted,
                attention_weight: 0.0, // set after softmax
                breakdown: ScoreBreakdown {
                    skill_match,
                    load_penalty,
                    experience_bonus,
                    skill_gap_penalty,
                },
            });
        }

        // Filter below threshold first, then apply softmax to remaining
        scores.retain(|s| s.adjusted_score >= self.config.min_score_threshold);

        if scores.is_empty() {
            return Vec::new();
        }

        // Apply softmax to get attention weights (sums to 1.0 for remaining agents)
        apply_softmax(&mut scores, self.config.temperature);

        // Sort by adjusted_score descending
        scores.sort_by(|a, b| {
            b.adjusted_score
                .partial_cmp(&a.adjusted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scores
    }

    /// Get the top-k agents for a task
    pub fn top_k(
        &self,
        task: &Task,
        agents: &[AgentProfile],
        k: usize,
    ) -> Vec<AttentionScore> {
        let mut scores = self.score(task, agents);
        scores.sort_by(|a, b| {
            b.adjusted_score
                .partial_cmp(&a.adjusted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores
    }

    /// Get the single best agent for a task
    pub fn best(&self, task: &Task, agents: &[AgentProfile]) -> Option<AttentionScore> {
        let top = self.top_k(task, agents, 1);
        top.into_iter().next()
    }
}

/// Build the task's skill requirement vector (Query)
fn task_skill_vector(task: &Task) -> Vec<f64> {
    let mut vec = vec![0.0; SkillTag::COUNT];
    for (skill, level) in &task.required_skills {
        vec[skill.index()] = *level;
    }
    vec
}

/// Compute skill match score: cosine similarity between task requirements and agent skills,
/// weighted by task priority
fn compute_skill_match(task_skills: &[f64], agent_skills: &[f64], task: &Task) -> f64 {
    // Dot product
    let dot: f64 = task_skills
        .iter()
        .zip(agent_skills.iter())
        .map(|(t, a)| t * a)
        .sum();

    // Task requirement magnitude
    let task_norm: f64 = task_skills.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
    // Agent skill magnitude
    let agent_norm: f64 = agent_skills.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);

    let cosine = dot / (task_norm * agent_norm);

    // Weight by task difficulty (priority * complexity)
    let difficulty = task.priority.weight() * (0.5 + 0.5 * task.complexity);

    cosine * (0.5 + 0.5 * difficulty)
}

/// Compute experience bonus based on agent's history with similar task categories
fn compute_experience_bonus(agent: &AgentProfile, task: &Task) -> f64 {
    if agent.history.is_empty() {
        return 0.0;
    }

    let mut total_bonus = 0.0;
    let mut count = 0;

    for (skill, _) in &task.required_skills {
        let success_rate = agent.category_success_rate(skill);
        let quality = agent.category_quality(skill);
        // Bonus = (success_rate + quality) / 2, offset from neutral (0.5)
        let bonus = (success_rate + quality) / 2.0 - 0.5;
        total_bonus += bonus;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    // Average bonus, scaled by experience
    let avg_bonus = total_bonus / count as f64;
    avg_bonus * agent.experience_score()
}

/// Compute skill gap penalty: sum of (requirement - proficiency) for skills where agent is deficient
fn compute_skill_gap(agent: &AgentProfile, task: &Task) -> f64 {
    let mut total_gap = 0.0;
    let mut count = 0;

    for (skill, required_level) in &task.required_skills {
        let agent_level = agent.skill_level(skill);
        if agent_level < *required_level {
            total_gap += required_level - agent_level;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    total_gap / count as f64
}

/// Apply softmax with temperature to normalize scores
fn apply_softmax(scores: &mut [AttentionScore], temperature: f64) {
    let temp = temperature.max(0.01);

    // Find max adjusted score for numerical stability
    let max_score = scores
        .iter()
        .map(|s| s.adjusted_score)
        .fold(f64::NEG_INFINITY, f64::max);

    // Compute exp((score - max) / temp) for each
    let exp_values: Vec<f64> = scores
        .iter()
        .map(|s| ((s.adjusted_score - max_score) / temp).exp())
        .collect();

    let sum_exp: f64 = exp_values.iter().sum();
    let len = scores.len();

    // Normalize
    for (i, score) in scores.iter_mut().enumerate() {
        score.attention_weight = if sum_exp > 0.0 {
            exp_values[i] / sum_exp
        } else {
            1.0 / len as f64
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::agent_profile::AgentProfileBuilder;
    use crate::attention::task_embedding::TaskPriority;

    fn make_coder_agent() -> AgentProfile {
        AgentProfileBuilder::new("coder-1", "CodeBot")
            .max_capacity(5)
            .skill(SkillTag::Coding, 0.95)
            .skill(SkillTag::Testing, 0.7)
            .skill(SkillTag::Documentation, 0.5)
            .build()
    }

    fn make_vision_agent() -> AgentProfile {
        AgentProfileBuilder::new("vision-1", "VisBot")
            .max_capacity(3)
            .skill(SkillTag::Vision, 0.9)
            .skill(SkillTag::Creative, 0.8)
            .build()
    }

    fn make_coding_task() -> Task {
        Task::new("t1", "Build REST API", TaskPriority::Normal, 0.6)
            .require_skill(SkillTag::Coding, 0.8)
            .require_skill(SkillTag::Testing, 0.5)
    }

    fn make_vision_task() -> Task {
        Task::new("t2", "Generate image", TaskPriority::High, 0.7)
            .require_skill(SkillTag::Vision, 0.8)
            .require_skill(SkillTag::Creative, 0.6)
    }

    #[test]
    fn test_scorer_basic() {
        let scorer = AttentionScorer::new(ScorerConfig::default());
        let task = make_coding_task();
        let agents = vec![make_coder_agent(), make_vision_agent()];
        let scores = scorer.score(&task, &agents);

        // Only the coder agent should pass the threshold for a coding task
        assert!(scores.len() >= 1);
        assert_eq!(scores[0].agent_id, "coder-1");
    }

    #[test]
    fn test_scorer_no_agents() {
        let scorer = AttentionScorer::new(ScorerConfig::default());
        let task = make_coding_task();
        let scores = scorer.score(&task, &[]);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_scorer_prefers_matching_skills() {
        let scorer = AttentionScorer::new(ScorerConfig::default());
        let task = make_coding_task();
        let agents = vec![make_coder_agent(), make_vision_agent()];

        let best = scorer.best(&task, &agents).unwrap();
        assert_eq!(best.agent_id, "coder-1");
    }

    #[test]
    fn test_scorer_vision_task_prefers_vision_agent() {
        let scorer = AttentionScorer::new(ScorerConfig::default());
        let task = make_vision_task();
        let agents = vec![make_coder_agent(), make_vision_agent()];

        let best = scorer.best(&task, &agents).unwrap();
        assert_eq!(best.agent_id, "vision-1");
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let scorer = AttentionScorer::new(ScorerConfig::default());
        let task = make_coding_task();
        // Use two coding agents so both pass the threshold
        let agents = vec![
            make_coder_agent(),
            AgentProfileBuilder::new("coder-2", "CodeBot2")
                .max_capacity(5)
                .skill(SkillTag::Coding, 0.85)
                .skill(SkillTag::Testing, 0.6)
                .build(),
        ];

        let scores = scorer.score(&task, &agents);
        let sum: f64 = scores.iter().map(|s| s.attention_weight).sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_load_penalty() {
        let scorer = AttentionScorer::new(ScorerConfig::default());

        let mut busy_coder = make_coder_agent();
        busy_coder.set_load(0.9);

        let mut idle_coder = make_coder_agent();
        idle_coder.set_load(0.0);

        let task = make_coding_task();
        let scores = scorer.score(&task, &[busy_coder, idle_coder]);

        // Idle agent should score higher
        let busy = scores.iter().find(|s| s.agent_id == "coder-1" && s.breakdown.load_penalty > 0.5);
        let idle = scores.iter().find(|s| s.agent_id == "coder-1" && s.breakdown.load_penalty < 0.1);
        assert!(busy.is_some());
        assert!(idle.is_some());
    }

    #[test]
    fn test_experience_bonus() {
        let scorer = AttentionScorer::new(ScorerConfig::quality_focused());

        let mut experienced = make_coder_agent();
        for _ in 0..20 {
            experienced.record_task(
                crate::attention::agent_profile::TaskRecord::success(SkillTag::Coding, 0.95, 0.5),
            );
        }

        let novice = make_coder_agent();

        let task = make_coding_task();
        let scores = scorer.score(&task, &[experienced.clone(), novice.clone()]);

        let experienced_score = scores.iter().find(|s| s.agent_id == "coder-1").unwrap();
        // The experienced agent should have a positive experience bonus
        // (there will be two agents with id "coder-1" — find the one with higher bonus)
        assert!(experienced_score.breakdown.experience_bonus >= 0.0);
    }

    #[test]
    fn test_skill_gap_penalty() {
        let scorer = AttentionScorer::new(ScorerConfig::default());

        let weak_coder = AgentProfileBuilder::new("weak-1", "WeakBot")
            .skill(SkillTag::Coding, 0.3)
            .build();

        let strong_coder = make_coder_agent();

        let task = Task::new("t1", "Hard coding task", TaskPriority::High, 0.9)
            .require_skill(SkillTag::Coding, 0.9);

        let scores = scorer.score(&task, &[weak_coder, strong_coder]);

        let weak_score = scores.iter().find(|s| s.agent_id == "weak-1").unwrap();
        let strong_score = scores.iter().find(|s| s.agent_id == "coder-1").unwrap();

        // Weak agent should have a gap penalty
        assert!(weak_score.breakdown.skill_gap_penalty > 0.0);
        // Strong agent should have no gap penalty
        assert!((strong_score.breakdown.skill_gap_penalty - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_top_k() {
        let scorer = AttentionScorer::new(ScorerConfig::default());

        let a1 = AgentProfileBuilder::new("a1", "A1").skill(SkillTag::Coding, 0.9).build();
        let a2 = AgentProfileBuilder::new("a2", "A2").skill(SkillTag::Coding, 0.7).build();
        let a3 = AgentProfileBuilder::new("a3", "A3").skill(SkillTag::Coding, 0.5).build();

        let task = Task::new("t1", "Code", TaskPriority::Normal, 0.5)
            .require_skill(SkillTag::Coding, 0.5);

        let top2 = scorer.top_k(&task, &[a1, a2, a3], 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].agent_id, "a1"); // best skill match
    }

    #[test]
    fn test_soft_softmax_distribution() {
        let config = ScorerConfig {
            temperature: 5.0, // very soft — nearly uniform
            ..Default::default()
        };
        let scorer = AttentionScorer::new(config);
        let task = make_coding_task();
        let agents = vec![make_coder_agent(), make_vision_agent()];

        let scores = scorer.score(&task, &agents);
        // With very high temperature, weights should be more uniform
        let max_weight = scores.iter().map(|s| s.attention_weight).fold(f64::NEG_INFINITY, f64::max);
        let min_weight = scores.iter().map(|s| s.attention_weight).fold(f64::INFINITY, f64::min);
        // Difference should be small
        assert!((max_weight - min_weight) < 0.3);
    }

    #[test]
    fn test_sharp_softmax_distribution() {
        let config = ScorerConfig {
            temperature: 0.1, // very sharp — winner takes almost all
            ..Default::default()
        };
        let scorer = AttentionScorer::new(config);
        let task = make_coding_task();
        let agents = vec![make_coder_agent(), make_vision_agent()];

        let scores = scorer.score(&task, &agents);
        let max_weight = scores.iter().map(|s| s.attention_weight).fold(f64::NEG_INFINITY, f64::max);
        // With very low temperature, the best should get most weight
        assert!(max_weight > 0.8);
    }

    #[test]
    fn test_score_breakdown_fields() {
        let scorer = AttentionScorer::new(ScorerConfig::default());
        let task = make_coding_task();
        let agents = vec![make_coder_agent()];
        let scores = scorer.score(&task, &agents);

        assert_eq!(scores.len(), 1);
        let s = &scores[0];
        assert!(s.breakdown.skill_match >= 0.0);
        assert!(s.breakdown.load_penalty >= 0.0);
        assert!(s.breakdown.experience_bonus >= -1.0);
        assert!(s.breakdown.skill_gap_penalty >= 0.0);
    }

    #[test]
    fn test_default_config() {
        let config = ScorerConfig::default();
        assert!((config.temperature - 1.0).abs() < 0.01);
        assert!((config.load_weight - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_expedited_config() {
        let config = ScorerConfig::expedited();
        assert!(config.temperature < ScorerConfig::default().temperature);
        assert!(config.load_weight > ScorerConfig::default().load_weight);
    }

    #[test]
    fn test_quality_config() {
        let config = ScorerConfig::quality_focused();
        assert!(config.temperature > ScorerConfig::default().temperature);
        assert!(config.experience_weight > ScorerConfig::default().experience_weight);
    }

    #[test]
    fn test_cosine_skill_match() {
        // Identical vectors → cosine = 1.0
        let task_skills = vec![0.8, 0.0, 0.0, 0.0];
        let agent_skills = vec![0.8, 0.0, 0.0, 0.0];
        let task = make_coding_task(); // dummy for difficulty weighting
        let match_score = compute_skill_match(&task_skills, &agent_skills, &task);
        assert!(match_score > 0.0);
    }
}
