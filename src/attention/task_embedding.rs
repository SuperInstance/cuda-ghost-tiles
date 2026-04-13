/*!
Task embedding for fleet task routing.

Tasks are embedded into fixed-dimension feature vectors that capture
skill requirements, priority, urgency, and complexity. These embeddings
serve as Query vectors in the attention-based routing mechanism.
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::agent_profile::SkillTag;

/// Task priority levels, ordered from lowest to highest urgency
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Low priority — best-effort, can be deferred
    Low = 0,
    /// Normal priority — standard processing
    Normal = 1,
    /// High priority — expedited processing
    High = 2,
    /// Critical priority — immediate processing required
    Critical = 3,
}

impl TaskPriority {
    /// Numeric weight for the priority (used in embeddings)
    pub fn weight(&self) -> f64 {
        match self {
            TaskPriority::Low => 0.25,
            TaskPriority::Normal => 0.50,
            TaskPriority::High => 0.75,
            TaskPriority::Critical => 1.0,
        }
    }

    /// Parse from string
    pub fn from_str_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "low" => Some(TaskPriority::Low),
            "normal" => Some(TaskPriority::Normal),
            "high" => Some(TaskPriority::High),
            "critical" => Some(TaskPriority::Critical),
            _ => None,
        }
    }
}

/// Describes the time pressure on a task
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskUrgency {
    /// Deadline timestamp in milliseconds (0 = no deadline)
    pub deadline_ms: u64,
    /// Time estimate in milliseconds for how long the task takes
    pub estimated_duration_ms: u64,
    /// Current timestamp when urgency was computed
    pub now_ms: u64,
}

impl TaskUrgency {
    /// Create urgency with a deadline offset from now
    pub fn with_deadline_from_now(deadline_offset_ms: u64, estimated_duration_ms: u64) -> Self {
        let now = current_ms();
        TaskUrgency {
            deadline_ms: now + deadline_offset_ms,
            estimated_duration_ms,
            now_ms: now,
        }
    }

    /// Create urgency with no deadline (low urgency)
    pub fn no_deadline(estimated_duration_ms: u64) -> Self {
        TaskUrgency {
            deadline_ms: 0,
            estimated_duration_ms,
            now_ms: current_ms(),
        }
    }

    /// Compute urgency score: 0.0 (no pressure) to 1.0 (extreme pressure).
    ///
    /// Based on the ratio of remaining time to estimated duration.
    /// If there's no deadline, urgency is 0.1 (baseline).
    pub fn urgency_score(&self) -> f64 {
        if self.deadline_ms == 0 {
            return 0.1; // low baseline urgency
        }
        let remaining = self.deadline_ms.saturating_sub(self.now_ms);
        if remaining == 0 {
            return 1.0; // past deadline
        }
        let ratio = remaining as f64 / self.estimated_duration_ms.max(1) as f64;
        // ratio > 2.0 → low urgency (< 0.2), ratio < 0.5 → high urgency (> 0.8)
        1.0 - ratio.clamp(0.0, 5.0) / 5.0
    }

    /// Whether the task is overdue
    pub fn is_overdue(&self) -> bool {
        self.deadline_ms > 0 && self.now_ms > self.deadline_ms
    }
}

/// A task in the fleet that needs to be routed to an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Required skills and minimum proficiency levels
    pub required_skills: HashMap<SkillTag, f64>,
    /// Task priority
    pub priority: TaskPriority,
    /// Urgency information
    pub urgency: TaskUrgency,
    /// Task complexity on a 0.0-1.0 scale
    pub complexity: f64,
    /// Cached embedding (computed lazily)
    embedding: Option<Vec<f64>>,
}

impl Task {
    /// Create a new task
    pub fn new(id: &str, description: &str, priority: TaskPriority, complexity: f64) -> Self {
        Task {
            id: id.to_string(),
            description: description.to_string(),
            required_skills: HashMap::new(),
            priority,
            urgency: TaskUrgency::no_deadline(60_000),
            complexity: complexity.clamp(0.0, 1.0),
            embedding: None,
        }
    }

    /// Add a required skill
    pub fn require_skill(mut self, skill: SkillTag, min_level: f64) -> Self {
        self.required_skills.insert(skill, min_level.clamp(0.0, 1.0));
        self
    }

    /// Set the urgency
    pub fn with_urgency(mut self, urgency: TaskUrgency) -> Self {
        self.urgency = urgency;
        self
    }

    /// Set estimated duration (convenience)
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.urgency.estimated_duration_ms = duration_ms;
        self
    }

    /// Set deadline (convenience)
    pub fn with_deadline(mut self, deadline_offset_ms: u64) -> Self {
        self.urgency = TaskUrgency::with_deadline_from_now(
            deadline_offset_ms,
            self.urgency.estimated_duration_ms,
        );
        self
    }

    /// Check if a task requires a specific skill
    pub fn requires_skill(&self, skill: &SkillTag) -> bool {
        self.required_skills.contains_key(skill)
    }

    /// Get the minimum required level for a skill
    pub fn required_level(&self, skill: &SkillTag) -> f64 {
        self.required_skills.get(skill).copied().unwrap_or(0.0)
    }

    /// How many skills does this task require?
    pub fn skill_count(&self) -> usize {
        self.required_skills.len()
    }

    /// Total "demand" of the task: sum of required skill levels
    pub fn total_demand(&self) -> f64 {
        self.required_skills.values().sum()
    }
}

/// Embeds tasks into fixed-dimension feature vectors.
///
/// The embedding dimension is structured as:
///   [0..SkillTag::COUNT] = required skill levels (the "query" of what's needed)
///   [SkillTag::COUNT] = priority weight
///   [SkillTag::COUNT + 1] = urgency score
///   [SkillTag::COUNT + 2] = complexity
///   [SkillTag::COUNT + 3] = task difficulty (priority * complexity * urgency)
pub struct TaskEmbedder {
    /// Dimension of the embedding vector
    pub embedding_dim: usize,
}

impl TaskEmbedder {
    pub fn new() -> Self {
        TaskEmbedder {
            embedding_dim: SkillTag::COUNT + 4,
        }
    }

    /// Embed a task into a feature vector.
    /// Returns a reference-counted slice to allow caching.
    pub fn embed<'a>(&self, task: &'a mut Task) -> &'a [f64] {
        if task.embedding.is_some() {
            return task.embedding.as_ref().unwrap();
        }

        let mut vec = vec![0.0; self.embedding_dim];

        // Skill requirements
        for (skill, min_level) in &task.required_skills {
            vec[skill.index()] = *min_level;
        }

        // Priority
        vec[SkillTag::COUNT] = task.priority.weight();

        // Urgency
        vec[SkillTag::COUNT + 1] = task.urgency.urgency_score();

        // Complexity
        vec[SkillTag::COUNT + 2] = task.complexity;

        // Composite difficulty
        vec[SkillTag::COUNT + 3] =
            task.priority.weight() * task.complexity * (0.5 + 0.5 * task.urgency.urgency_score());

        task.embedding = Some(vec);
        task.embedding.as_ref().unwrap()
    }

    /// Compute embedding without mutating the task
    pub fn compute_embedding(&self, task: &Task) -> Vec<f64> {
        let mut task_clone = task.clone();
        self.embed(&mut task_clone).to_vec()
    }
}

fn current_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_priority_weight() {
        assert!(TaskPriority::Low.weight() < TaskPriority::Normal.weight());
        assert!(TaskPriority::Normal.weight() < TaskPriority::High.weight());
        assert!(TaskPriority::High.weight() < TaskPriority::Critical.weight());
        assert!((TaskPriority::Critical.weight() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_task_priority_from_str() {
        assert_eq!(TaskPriority::from_str_label("low"), Some(TaskPriority::Low));
        assert_eq!(TaskPriority::from_str_label("HIGH"), Some(TaskPriority::High));
        assert_eq!(TaskPriority::from_str_label("unknown"), None);
    }

    #[test]
    fn test_urgency_no_deadline() {
        let u = TaskUrgency::no_deadline(60_000);
        assert!((u.urgency_score() - 0.1).abs() < 0.01);
        assert!(!u.is_overdue());
    }

    #[test]
    fn test_urgency_with_deadline() {
        // Deadline 5 seconds from now, task takes 10 seconds → tight
        let u = TaskUrgency::with_deadline_from_now(5_000, 10_000);
        assert!(u.urgency_score() > 0.5);
    }

    #[test]
    fn test_urgency_comfortable() {
        // Deadline 60 seconds from now, task takes 5 seconds → comfortable
        let u = TaskUrgency::with_deadline_from_now(60_000, 5_000);
        assert!(u.urgency_score() < 0.5);
    }

    #[test]
    fn test_task_new() {
        let t = Task::new("t1", "Build API", TaskPriority::Normal, 0.6);
        assert_eq!(t.id, "t1");
        assert_eq!(t.priority, TaskPriority::Normal);
        assert!((t.complexity - 0.6).abs() < 0.01);
        assert!(t.required_skills.is_empty());
    }

    #[test]
    fn test_task_require_skill() {
        let t = Task::new("t1", "Build API", TaskPriority::Normal, 0.6)
            .require_skill(SkillTag::Coding, 0.8)
            .require_skill(SkillTag::Testing, 0.5);
        assert!(t.requires_skill(&SkillTag::Coding));
        assert!(!t.requires_skill(&SkillTag::Vision));
        assert!((t.required_level(&SkillTag::Coding) - 0.8).abs() < 0.01);
        assert_eq!(t.skill_count(), 2);
    }

    #[test]
    fn test_task_complexity_clamping() {
        let t = Task::new("t1", "Task", TaskPriority::Normal, 1.5);
        assert!((t.complexity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_task_total_demand() {
        let t = Task::new("t1", "Task", TaskPriority::Normal, 0.5)
            .require_skill(SkillTag::Coding, 0.7)
            .require_skill(SkillTag::Testing, 0.3);
        assert!((t.total_demand() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_embedder_dimension() {
        let e = TaskEmbedder::new();
        assert_eq!(e.embedding_dim, SkillTag::COUNT + 4);
    }

    #[test]
    fn test_task_embedding() {
        let mut e = TaskEmbedder::new();
        let mut t = Task::new("t1", "Build API", TaskPriority::High, 0.7)
            .require_skill(SkillTag::Coding, 0.9)
            .require_skill(SkillTag::Testing, 0.6);

        let embedding = e.embed(&mut t);
        assert_eq!(embedding.len(), e.embedding_dim);
        assert!((embedding[SkillTag::Coding.index()] - 0.9).abs() < 0.01);
        assert!((embedding[SkillTag::Testing.index()] - 0.6).abs() < 0.01);
        assert!((embedding[SkillTag::Vision.index()] - 0.0).abs() < 0.01);
        // Priority weight for High
        assert!((embedding[SkillTag::COUNT] - 0.75).abs() < 0.01);
        // Urgency (no deadline → 0.1)
        assert!((embedding[SkillTag::COUNT + 1] - 0.1).abs() < 0.01);
        // Complexity
        assert!((embedding[SkillTag::COUNT + 2] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_embedding_caching() {
        let embedder = TaskEmbedder::new();
        let mut t = Task::new("t1", "Task", TaskPriority::Normal, 0.5);
        let first = embedder.embed(&mut t).as_ptr();
        let second = embedder.embed(&mut t).as_ptr();
        // Same pointer → cached
        assert_eq!(first, second);
    }

    #[test]
    fn test_compute_embedding_no_mutate() {
        let embedder = TaskEmbedder::new();
        let t = Task::new("t1", "Task", TaskPriority::Normal, 0.5)
            .require_skill(SkillTag::Coding, 0.8);
        let embedding = embedder.compute_embedding(&t);
        assert_eq!(embedding.len(), embedder.embedding_dim);
        // Original task embedding should still be None
        assert!(t.embedding.is_none());
    }

    #[test]
    fn test_composite_difficulty() {
        let mut e = TaskEmbedder::new();
        // High priority, high complexity, high urgency
        let mut t = Task::new("t1", "Critical task", TaskPriority::Critical, 0.9)
            .with_deadline(1_000); // very tight deadline
        let emb = e.embed(&mut t);
        let difficulty = emb[SkillTag::COUNT + 3];
        // Should be high
        assert!(difficulty > 0.5);
    }
}
