/*!
Agent capability profiles for fleet task routing.

Each agent has a profile capturing its skills, current load, and historical
performance. Profiles are embedded into feature vectors for attention-based
matching against task requirements.
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for an agent in the fleet
pub type AgentId = String;

/// Skill tags that agents can possess and tasks can require.
/// Covers the main domains a fleet agent might operate in.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SkillTag {
    /// Code generation, review, debugging
    Coding,
    /// Natural language understanding and generation
    Language,
    /// Data analysis and visualization
    DataAnalysis,
    /// Mathematical reasoning and computation
    Mathematics,
    /// Image/video understanding and generation
    Vision,
    /// Web browsing and research
    WebResearch,
    /// Database operations and queries
    Database,
    /// System administration and DevOps
    DevOps,
    /// Creative writing and design
    Creative,
    /// Testing and quality assurance
    Testing,
    /// Security analysis and auditing
    Security,
    /// Documentation
    Documentation,
}

impl SkillTag {
    /// Total number of known skill tags (used for embedding dimension)
    pub const COUNT: usize = 12;

    /// Convert to a canonical string label
    pub fn as_str(&self) -> &'static str {
        match self {
            SkillTag::Coding => "coding",
            SkillTag::Language => "language",
            SkillTag::DataAnalysis => "data_analysis",
            SkillTag::Mathematics => "mathematics",
            SkillTag::Vision => "vision",
            SkillTag::WebResearch => "web_research",
            SkillTag::Database => "database",
            SkillTag::DevOps => "devops",
            SkillTag::Creative => "creative",
            SkillTag::Testing => "testing",
            SkillTag::Security => "security",
            SkillTag::Documentation => "documentation",
        }
    }

    /// Parse a skill tag from string
    pub fn from_str_label(s: &str) -> Option<Self> {
        match s {
            "coding" => Some(SkillTag::Coding),
            "language" => Some(SkillTag::Language),
            "data_analysis" => Some(SkillTag::DataAnalysis),
            "mathematics" => Some(SkillTag::Mathematics),
            "vision" => Some(SkillTag::Vision),
            "web_research" => Some(SkillTag::WebResearch),
            "database" => Some(SkillTag::Database),
            "devops" => Some(SkillTag::DevOps),
            "creative" => Some(SkillTag::Creative),
            "testing" => Some(SkillTag::Testing),
            "security" => Some(SkillTag::Security),
            "documentation" => Some(SkillTag::Documentation),
            _ => None,
        }
    }

    /// Get index in the canonical skill ordering (for embedding)
    pub fn index(&self) -> usize {
        match self {
            SkillTag::Coding => 0,
            SkillTag::Language => 1,
            SkillTag::DataAnalysis => 2,
            SkillTag::Mathematics => 3,
            SkillTag::Vision => 4,
            SkillTag::WebResearch => 5,
            SkillTag::Database => 6,
            SkillTag::DevOps => 7,
            SkillTag::Creative => 8,
            SkillTag::Testing => 9,
            SkillTag::Security => 10,
            SkillTag::Documentation => 11,
        }
    }

    /// Iterate over all skill tags in canonical order
    pub fn all() -> impl Iterator<Item = SkillTag> {
        [
            SkillTag::Coding,
            SkillTag::Language,
            SkillTag::DataAnalysis,
            SkillTag::Mathematics,
            SkillTag::Vision,
            SkillTag::WebResearch,
            SkillTag::Database,
            SkillTag::DevOps,
            SkillTag::Creative,
            SkillTag::Testing,
            SkillTag::Security,
            SkillTag::Documentation,
        ]
        .into_iter()
    }
}

/// A record of a task the agent has completed (or failed)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskRecord {
    /// Category of the task (matches SkillTag)
    pub category: SkillTag,
    /// Whether the task succeeded
    pub success: bool,
    /// Quality score 0.0-1.0 assigned by the evaluator
    pub quality: f64,
    /// Complexity of the task
    pub complexity: f64,
    /// Timestamp in millis
    pub completed_at_ms: u64,
}

impl TaskRecord {
    pub fn success(category: SkillTag, quality: f64, complexity: f64) -> Self {
        TaskRecord {
            category,
            success: true,
            quality: quality.clamp(0.0, 1.0),
            complexity,
            completed_at_ms: now_ms(),
        }
    }

    pub fn failure(category: SkillTag, complexity: f64) -> Self {
        TaskRecord {
            category,
            success: false,
            quality: 0.0,
            complexity,
            completed_at_ms: now_ms(),
        }
    }
}

/// Profile of a fleet agent, tracking capabilities, load, and history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentProfile {
    /// Unique agent identifier
    pub id: AgentId,
    /// Human-readable agent name
    pub name: String,
    /// Skills and proficiency levels (0.0 = none, 1.0 = expert)
    pub skills: HashMap<SkillTag, f64>,
    /// Current workload as fraction of max capacity (0.0 = idle, 1.0 = full)
    pub current_load: f64,
    /// Maximum capacity (number of concurrent tasks)
    pub max_capacity: u32,
    /// Historical task records (most recent last)
    pub history: Vec<TaskRecord>,
    /// Maximum history length to retain
    pub max_history: usize,
    /// When the agent was registered
    pub registered_at_ms: u64,
    /// Whether the agent is online and available
    pub online: bool,
}

impl AgentProfile {
    /// Create a new agent profile
    pub fn new(id: &str, name: &str, max_capacity: u32) -> Self {
        AgentProfile {
            id: id.to_string(),
            name: name.to_string(),
            skills: HashMap::new(),
            current_load: 0.0,
            max_capacity,
            history: Vec::new(),
            max_history: 100,
            registered_at_ms: now_ms(),
            online: true,
        }
    }

    /// Set a skill proficiency level
    pub fn set_skill(&mut self, skill: SkillTag, proficiency: f64) -> &mut Self {
        self.skills.insert(skill, proficiency.clamp(0.0, 1.0));
        self
    }

    /// Get the proficiency for a skill (0.0 if not possessed)
    pub fn skill_level(&self, skill: &SkillTag) -> f64 {
        self.skills.get(skill).copied().unwrap_or(0.0)
    }

    /// Check if the agent has a skill above a threshold
    pub fn has_skill(&self, skill: &SkillTag, min_level: f64) -> bool {
        self.skill_level(skill) >= min_level
    }

    /// Number of active concurrent tasks this agent can still accept
    pub fn available_slots(&self) -> u32 {
        let active = (self.current_load * self.max_capacity as f64).round() as u32;
        self.max_capacity.saturating_sub(active)
    }

    /// Whether this agent can accept more tasks
    pub fn is_available(&self) -> bool {
        self.online && self.available_slots() > 0
    }

    /// Update the agent's load (accepts value in 0.0-1.0)
    pub fn set_load(&mut self, load: f64) {
        self.current_load = load.clamp(0.0, 1.0);
    }

    /// Record a completed task
    pub fn record_task(&mut self, record: TaskRecord) {
        self.history.push(record);
        // Trim history if needed
        if self.history.len() > self.max_history {
            let excess = self.history.len() - self.max_history;
            self.history.drain(0..excess);
        }
    }

    /// Overall success rate across all history
    pub fn success_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.5; // neutral prior
        }
        let successes = self.history.iter().filter(|r| r.success).count();
        successes as f64 / self.history.len() as f64
    }

    /// Success rate for tasks in a specific category
    pub fn category_success_rate(&self, category: &SkillTag) -> f64 {
        let relevant: Vec<_> = self.history.iter().filter(|r| &r.category == category).collect();
        if relevant.is_empty() {
            return 0.5; // neutral prior
        }
        let successes = relevant.iter().filter(|r| r.success).count();
        successes as f64 / relevant.len() as f64
    }

    /// Average quality score for successful tasks in a category
    pub fn category_quality(&self, category: &SkillTag) -> f64 {
        let relevant: Vec<_> = self
            .history
            .iter()
            .filter(|r| &r.category == category && r.success)
            .collect();
        if relevant.is_empty() {
            return 0.5; // neutral prior
        }
        let sum: f64 = relevant.iter().map(|r| r.quality).sum();
        sum / relevant.len() as f64
    }

    /// Aggregate experience score: how experienced is this agent?
    pub fn experience_score(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        // Logarithmic scaling: 10 tasks ≈ 0.5, 100 tasks ≈ 0.75, 1000 tasks ≈ 1.0
        1.0 - (-(self.history.len() as f64) / 300.0).exp()
    }

    /// Generate a skill vector of fixed size (SkillTag::COUNT) for attention embedding.
    /// Each position corresponds to a SkillTag's proficiency.
    pub fn skill_vector(&self) -> Vec<f64> {
        let mut vec = vec![0.0; SkillTag::COUNT];
        for (skill, proficiency) in &self.skills {
            vec[skill.index()] = *proficiency;
        }
        vec
    }

    /// Generate a performance vector of fixed size for attention embedding.
    /// Each position encodes (success_rate + quality) / 2 for that skill category.
    pub fn performance_vector(&self) -> Vec<f64> {
        SkillTag::all()
            .map(|skill| {
                let sr = self.category_success_rate(&skill);
                let q = self.category_quality(&skill);
                (sr + q) / 2.0
            })
            .collect()
    }
}

/// Builder for constructing AgentProfile fluently
pub struct AgentProfileBuilder {
    id: String,
    name: String,
    max_capacity: u32,
    skills: HashMap<SkillTag, f64>,
    max_history: usize,
}

impl AgentProfileBuilder {
    pub fn new(id: &str, name: &str) -> Self {
        AgentProfileBuilder {
            id: id.to_string(),
            name: name.to_string(),
            max_capacity: 5,
            skills: HashMap::new(),
            max_history: 100,
        }
    }

    pub fn max_capacity(mut self, cap: u32) -> Self {
        self.max_capacity = cap;
        self
    }

    pub fn skill(mut self, skill: SkillTag, level: f64) -> Self {
        self.skills.insert(skill, level.clamp(0.0, 1.0));
        self
    }

    pub fn max_history(mut self, size: usize) -> Self {
        self.max_history = size;
        self
    }

    pub fn build(self) -> AgentProfile {
        AgentProfile {
            id: self.id,
            name: self.name,
            skills: self.skills,
            current_load: 0.0,
            max_capacity: self.max_capacity,
            history: Vec::new(),
            max_history: self.max_history,
            registered_at_ms: now_ms(),
            online: true,
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_tag_all_count() {
        let all: Vec<_> = SkillTag::all().collect();
        assert_eq!(all.len(), SkillTag::COUNT);
    }

    #[test]
    fn test_skill_tag_index_unique() {
        let indices: std::collections::HashSet<usize> = SkillTag::all().map(|s| s.index()).collect();
        assert_eq!(indices.len(), SkillTag::COUNT);
    }

    #[test]
    fn test_skill_tag_from_str_label() {
        assert_eq!(SkillTag::from_str_label("coding"), Some(SkillTag::Coding));
        assert_eq!(SkillTag::from_str_label("unknown"), None);
    }

    #[test]
    fn test_agent_profile_new() {
        let p = AgentProfile::new("a1", "Alpha", 5);
        assert_eq!(p.id, "a1");
        assert_eq!(p.name, "Alpha");
        assert_eq!(p.max_capacity, 5);
        assert!(p.skills.is_empty());
        assert!(p.is_available());
    }

    #[test]
    fn test_set_and_get_skill() {
        let mut p = AgentProfile::new("a1", "Alpha", 5);
        p.set_skill(SkillTag::Coding, 0.9);
        assert_eq!(p.skill_level(&SkillTag::Coding), 0.9);
        assert_eq!(p.skill_level(&SkillTag::Vision), 0.0);
    }

    #[test]
    fn test_has_skill() {
        let mut p = AgentProfile::new("a1", "Alpha", 5);
        p.set_skill(SkillTag::Coding, 0.8);
        assert!(p.has_skill(&SkillTag::Coding, 0.7));
        assert!(!p.has_skill(&SkillTag::Coding, 0.9));
        assert!(!p.has_skill(&SkillTag::Vision, 0.1));
    }

    #[test]
    fn test_available_slots() {
        let mut p = AgentProfile::new("a1", "Alpha", 5);
        assert_eq!(p.available_slots(), 5);
        p.set_load(0.6); // 3 tasks active
        assert_eq!(p.available_slots(), 2);
        p.set_load(1.0); // full
        assert_eq!(p.available_slots(), 0);
        assert!(!p.is_available());
    }

    #[test]
    fn test_online_offline() {
        let mut p = AgentProfile::new("a1", "Alpha", 5);
        assert!(p.is_available());
        p.online = false;
        assert!(!p.is_available());
    }

    #[test]
    fn test_record_task_and_success_rate() {
        let mut p = AgentProfile::new("a1", "Alpha", 5);
        // No history → neutral prior
        assert!((p.success_rate() - 0.5).abs() < 0.01);

        p.record_task(TaskRecord::success(SkillTag::Coding, 0.9, 0.5));
        p.record_task(TaskRecord::success(SkillTag::Coding, 0.8, 0.5));
        p.record_task(TaskRecord::failure(SkillTag::Coding, 0.5));
        assert!((p.success_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_category_success_rate() {
        let mut p = AgentProfile::new("a1", "Alpha", 5);
        p.record_task(TaskRecord::success(SkillTag::Coding, 0.9, 0.5));
        p.record_task(TaskRecord::failure(SkillTag::Vision, 0.5));
        assert!((p.category_success_rate(&SkillTag::Coding) - 1.0).abs() < 0.01);
        assert!((p.category_success_rate(&SkillTag::Vision) - 0.0).abs() < 0.01);
        // No history for Math → neutral
        assert!((p.category_success_rate(&SkillTag::Mathematics) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_category_quality() {
        let mut p = AgentProfile::new("a1", "Alpha", 5);
        p.record_task(TaskRecord::success(SkillTag::Coding, 0.8, 0.5));
        p.record_task(TaskRecord::success(SkillTag::Coding, 0.4, 0.5));
        assert!((p.category_quality(&SkillTag::Coding) - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_experience_score() {
        let p = AgentProfile::new("a1", "Alpha", 5);
        assert!((p.experience_score() - 0.0).abs() < 0.01);

        let mut p = AgentProfile::new("a1", "Alpha", 5);
        for _ in 0..10 {
            p.record_task(TaskRecord::success(SkillTag::Coding, 0.5, 0.5));
        }
        assert!(p.experience_score() > 0.0);
    }

    #[test]
    fn test_skill_vector() {
        let mut p = AgentProfile::new("a1", "Alpha", 5);
        p.set_skill(SkillTag::Coding, 0.9);
        p.set_skill(SkillTag::Vision, 0.3);
        let sv = p.skill_vector();
        assert_eq!(sv.len(), SkillTag::COUNT);
        assert!((sv[SkillTag::Coding.index()] - 0.9).abs() < 0.01);
        assert!((sv[SkillTag::Vision.index()] - 0.3).abs() < 0.01);
        assert!((sv[SkillTag::Mathematics.index()] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_performance_vector() {
        let p = AgentProfile::new("a1", "Alpha", 5);
        let pv = p.performance_vector();
        assert_eq!(pv.len(), SkillTag::COUNT);
        // All neutral priors (0.5 + 0.5) / 2 = 0.5
        for v in &pv {
            assert!((v - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_history_trimming() {
        let mut p = AgentProfile::new("a1", "Alpha", 5);
        p.max_history = 5;
        for _ in 0..10 {
            p.record_task(TaskRecord::success(SkillTag::Coding, 0.5, 0.5));
        }
        assert_eq!(p.history.len(), 5);
    }

    #[test]
    fn test_builder() {
        let p = AgentProfileBuilder::new("a1", "Alpha")
            .max_capacity(10)
            .skill(SkillTag::Coding, 0.95)
            .skill(SkillTag::Mathematics, 0.8)
            .max_history(200)
            .build();

        assert_eq!(p.id, "a1");
        assert_eq!(p.max_capacity, 10);
        assert_eq!(p.skill_level(&SkillTag::Coding), 0.95);
        assert_eq!(p.skill_level(&SkillTag::Mathematics), 0.8);
        assert_eq!(p.max_history, 200);
    }

    #[test]
    fn test_task_record_constructors() {
        let s = TaskRecord::success(SkillTag::Coding, 0.8, 0.6);
        assert!(s.success);
        assert_eq!(s.category, SkillTag::Coding);

        let f = TaskRecord::failure(SkillTag::Vision, 0.4);
        assert!(!f.success);
        assert_eq!(f.quality, 0.0);
    }
}
