/*!
# Fleet Task Routing Attention Mechanism

An attention-based system for routing tasks to agents in a fleet, inspired by
transformer attention but adapted for agent-task matching.

## Architecture

```text
Task -> TaskEmbedder -> Query (Q)
                              |
Agent Profile -> Skill/Perf Vectors -> Key (K)
                              |
                         AttentionScorer
                         (Q*K^T / sqrt(d_k))
                              |
                         [Load Penalty, Experience Bonus]
                              |
                         Softmax -> Weights
                              |
                         TaskRouter
                              |
                       RoutingDecision
```

## Components

- **AgentProfile**: Captures agent skills, load, and history
- **Task**: Describes what needs to be done (skills, priority, urgency)
- **TaskEmbedder**: Converts tasks to feature vectors (Query)
- **AttentionScorer**: Computes Q/K/V attention scores with modifiers
- **TaskRouter**: Makes final routing decisions using various strategies
- **AgentPool**: Manages the fleet of agents (register, remove, update)

## Usage

```ignore
use cuda_ghost_tiles::attention::{AgentPool, TaskRouter, TaskEmbedder};
use cuda_ghost_tiles::attention::agent_profile::{AgentProfileBuilder, SkillTag};
use cuda_ghost_tiles::attention::task_embedding::{Task, TaskPriority};

// Build the agent pool
let mut pool = AgentPool::new();
pool.register(AgentProfileBuilder::new("coder", "CodeBot")
    .max_capacity(5)
    .skill(SkillTag::Coding, 0.95)
    .build());

// Create a task and embed it
let mut task = Task::new("t1", "Build API", TaskPriority::High, 0.7)
    .require_skill(SkillTag::Coding, 0.8);
let embedder = TaskEmbedder::new();
embedder.embed(&mut task);

// Route the task
let mut router = TaskRouter::with_defaults();
let agents: Vec<_> = pool.available().into_iter().cloned().collect();
let decision = router.route(&task, &agents);
```
*/

pub mod agent_profile;
pub mod task_embedding;
pub mod scorer;
pub mod router;
pub mod pool;

// Re-exports for convenient access
pub use agent_profile::{AgentId, AgentProfile, AgentProfileBuilder, SkillTag, TaskRecord};
pub use task_embedding::{Task, TaskEmbedder, TaskPriority, TaskUrgency};
pub use scorer::{AttentionScore, AttentionScorer, ScoreBreakdown, ScorerConfig};
pub use router::{RoutingDecision, RoutingStrategy, RoutingStats, TaskRouter};
pub use pool::{AgentPool, AgentPoolBuilder, PoolEvent, PoolStats};
