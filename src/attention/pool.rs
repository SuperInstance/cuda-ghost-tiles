/*!
Agent pool management for fleet task routing.

Provides lifecycle management for agents: registration, removal,
status updates, and queries. The pool is the primary interface
for the router to access available agents.
*/

use super::agent_profile::{AgentId, AgentProfile, SkillTag};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Event types emitted by the agent pool
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PoolEvent {
    /// A new agent was registered
    AgentRegistered(AgentId),
    /// An agent was removed from the pool
    AgentRemoved(AgentId),
    /// An agent's profile was updated
    AgentUpdated(AgentId),
    /// An agent went online
    AgentOnline(AgentId),
    /// An agent went offline
    AgentOffline(AgentId),
}

/// Pool statistics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PoolStats {
    /// Total agents registered
    pub total_agents: usize,
    /// Agents currently online
    pub online_agents: usize,
    /// Agents currently available (online + has capacity)
    pub available_agents: usize,
    /// Average load across all online agents
    pub avg_load: f64,
    /// Total available slots across all online agents
    pub total_available_slots: u32,
    /// Skill coverage: fraction of SkillTags covered by at least one agent
    pub skill_coverage: f64,
}

/// Manages the fleet of agents with registration, removal, and status tracking
pub struct AgentPool {
    /// All registered agents keyed by ID
    agents: HashMap<AgentId, AgentProfile>,
    /// Event log
    events: Vec<PoolEvent>,
    /// Maximum event log size
    max_events: usize,
}

impl AgentPool {
    /// Create an empty agent pool
    pub fn new() -> Self {
        AgentPool {
            agents: HashMap::new(),
            events: Vec::new(),
            max_events: 1000,
        }
    }

    /// Register a new agent in the pool
    pub fn register(&mut self, profile: AgentProfile) -> bool {
        let id = profile.id.clone();
        if self.agents.contains_key(&id) {
            return false; // already registered
        }
        self.emit(PoolEvent::AgentRegistered(id.clone()));
        self.agents.insert(id, profile);
        true
    }

    /// Remove an agent from the pool
    pub fn remove(&mut self, agent_id: &str) -> bool {
        if self.agents.remove(agent_id).is_some() {
            self.emit(PoolEvent::AgentRemoved(agent_id.to_string()));
            return true;
        }
        false
    }

    /// Get a reference to an agent's profile
    pub fn get(&self, agent_id: &str) -> Option<&AgentProfile> {
        self.agents.get(agent_id)
    }

    /// Get a mutable reference to an agent's profile
    pub fn get_mut(&mut self, agent_id: &str) -> Option<&mut AgentProfile> {
        self.agents.get_mut(agent_id)
    }

    /// Set an agent's online status
    pub fn set_online(&mut self, agent_id: &str, online: bool) -> bool {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.online = online;
            if online {
                self.emit(PoolEvent::AgentOnline(agent_id.to_string()));
            } else {
                self.emit(PoolEvent::AgentOffline(agent_id.to_string()));
            }
            return true;
        }
        false
    }

    /// Update an agent's load
    pub fn set_load(&mut self, agent_id: &str, load: f64) -> bool {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.set_load(load);
            self.emit(PoolEvent::AgentUpdated(agent_id.to_string()));
            return true;
        }
        false
    }

    /// Update an agent's skills
    pub fn update_skills(
        &mut self,
        agent_id: &str,
        skills: HashMap<SkillTag, f64>,
    ) -> bool {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            for (skill, level) in skills {
                agent.set_skill(skill, level);
            }
            self.emit(PoolEvent::AgentUpdated(agent_id.to_string()));
            return true;
        }
        false
    }

    /// Record a task result for an agent
    pub fn record_result(
        &mut self,
        agent_id: &str,
        record: super::agent_profile::TaskRecord,
    ) -> bool {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.record_task(record);
            self.emit(PoolEvent::AgentUpdated(agent_id.to_string()));
            return true;
        }
        false
    }

    /// Get all registered agents
    pub fn all(&self) -> Vec<&AgentProfile> {
        self.agents.values().collect()
    }

    /// Get all available agents (online + has capacity)
    pub fn available(&self) -> Vec<&AgentProfile> {
        self.agents.values().filter(|a| a.is_available()).collect()
    }

    /// Get all online agents
    pub fn online(&self) -> Vec<&AgentProfile> {
        self.agents.values().filter(|a| a.online).collect()
    }

    /// Find agents that have a specific skill above a minimum level
    pub fn find_by_skill(&self, skill: &SkillTag, min_level: f64) -> Vec<&AgentProfile> {
        self.agents
            .values()
            .filter(|a| a.has_skill(skill, min_level))
            .collect()
    }

    /// Find agents whose skills overlap with a task's requirements
    pub fn find_qualified(
        &self,
        required_skills: &HashMap<SkillTag, f64>,
    ) -> Vec<&AgentProfile> {
        self.agents
            .values()
            .filter(|agent| {
                required_skills.iter().all(|(skill, min_level)| {
                    agent.skill_level(skill) >= *min_level
                })
            })
            .collect()
    }

    /// Number of registered agents
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Whether the pool is empty
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Get the event log
    pub fn events(&self) -> &[PoolEvent] {
        &self.events
    }

    /// Clear the event log
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Compute pool statistics
    pub fn stats(&self) -> PoolStats {
        let total = self.agents.len();
        let online_count = self.agents.values().filter(|a| a.online).count();
        let available: Vec<&AgentProfile> = self.available();
        let available_count = available.len();
        let avg_load = if online_count > 0 {
            self.agents
                .values()
                .filter(|a| a.online)
                .map(|a| a.current_load)
                .sum::<f64>()
                / online_count as f64
        } else {
            0.0
        };
        let total_slots: u32 = available.iter().map(|a| a.available_slots()).sum();

        // Skill coverage: how many of the 12 SkillTags are covered?
        let covered_skills: std::collections::HashSet<SkillTag> = self
            .agents
            .values()
            .flat_map(|a| a.skills.keys().cloned())
            .collect();
        let skill_coverage = covered_skills.len() as f64 / SkillTag::COUNT as f64;

        PoolStats {
            total_agents: total,
            online_agents: online_count,
            available_agents: available_count,
            avg_load,
            total_available_slots: total_slots,
            skill_coverage,
        }
    }

    /// Emit a pool event
    fn emit(&mut self, event: PoolEvent) {
        self.events.push(event);
        if self.events.len() > self.max_events {
            let excess = self.events.len() - self.max_events;
            self.events.drain(0..excess);
        }
    }
}

impl Default for AgentPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating pre-populated agent pools
pub struct AgentPoolBuilder {
    pool: AgentPool,
}

impl AgentPoolBuilder {
    pub fn new() -> Self {
        AgentPoolBuilder {
            pool: AgentPool::new(),
        }
    }

    /// Add an agent to the pool
    pub fn agent(mut self, profile: AgentProfile) -> Self {
        self.pool.register(profile);
        self
    }

    /// Add an agent using the builder pattern
    pub fn add(self, id: &str, name: &str) -> AgentInPoolBuilder {
        AgentInPoolBuilder {
            pool_builder: self,
            profile: AgentProfile::new(id, name, 5),
        }
    }

    pub fn build(self) -> AgentPool {
        self.pool
    }
}

/// Helper builder for adding agents to a pool
pub struct AgentInPoolBuilder {
    pool_builder: AgentPoolBuilder,
    profile: AgentProfile,
}

impl AgentInPoolBuilder {
    pub fn capacity(mut self, cap: u32) -> Self {
        self.profile.max_capacity = cap;
        self
    }

    pub fn skill(mut self, skill: SkillTag, level: f64) -> Self {
        self.profile.set_skill(skill, level);
        self
    }

    pub fn offline(mut self) -> Self {
        self.profile.online = false;
        self
    }

    pub fn load(mut self, load: f64) -> Self {
        self.profile.set_load(load);
        self
    }

    pub fn done(mut self) -> AgentPoolBuilder {
        self.pool_builder.pool.register(self.profile);
        self.pool_builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::agent_profile::TaskRecord;

    fn make_sample_pool() -> AgentPool {
        AgentPoolBuilder::new()
            .add("coder", "CodeBot")
                .capacity(5)
                .skill(SkillTag::Coding, 0.95)
                .skill(SkillTag::Testing, 0.7)
                .done()
            .add("vision", "VisBot")
                .capacity(3)
                .skill(SkillTag::Vision, 0.9)
                .skill(SkillTag::Creative, 0.8)
                .done()
            .add("math", "MathBot")
                .capacity(2)
                .skill(SkillTag::Mathematics, 0.95)
                .done()
            .build()
    }

    #[test]
    fn test_pool_register() {
        let mut pool = AgentPool::new();
        let agent = AgentProfile::new("a1", "Alpha", 5);
        assert!(pool.register(agent));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_pool_register_duplicate() {
        let mut pool = AgentPool::new();
        let a1 = AgentProfile::new("a1", "Alpha", 5);
        let a2 = AgentProfile::new("a1", "Alpha v2", 5);
        assert!(pool.register(a1));
        assert!(!pool.register(a2)); // duplicate
    }

    #[test]
    fn test_pool_remove() {
        let mut pool = make_sample_pool();
        assert!(pool.remove("coder"));
        assert_eq!(pool.len(), 2);
        assert!(!pool.remove("nonexistent"));
    }

    #[test]
    fn test_pool_get() {
        let pool = make_sample_pool();
        let agent = pool.get("coder").unwrap();
        assert_eq!(agent.name, "CodeBot");
        assert!(pool.get("nonexistent").is_none());
    }

    #[test]
    fn test_pool_get_mut() {
        let mut pool = make_sample_pool();
        if let Some(agent) = pool.get_mut("coder") {
            agent.set_skill(SkillTag::Security, 0.5);
        }
        assert!(pool.get("coder").unwrap().has_skill(&SkillTag::Security, 0.4));
    }

    #[test]
    fn test_pool_set_online() {
        let mut pool = make_sample_pool();
        assert!(pool.set_online("coder", false));
        assert!(!pool.get("coder").unwrap().online);
        assert!(!pool.set_online("nonexistent", true));
    }

    #[test]
    fn test_pool_set_load() {
        let mut pool = make_sample_pool();
        assert!(pool.set_load("coder", 0.8));
        assert!((pool.get("coder").unwrap().current_load - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_pool_update_skills() {
        let mut pool = make_sample_pool();
        let mut new_skills = HashMap::new();
        new_skills.insert(SkillTag::Security, 0.7);
        assert!(pool.update_skills("coder", new_skills));
        assert!((pool.get("coder").unwrap().skill_level(&SkillTag::Security) - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_pool_record_result() {
        let mut pool = make_sample_pool();
        let record = TaskRecord::success(SkillTag::Coding, 0.9, 0.5);
        assert!(pool.record_result("coder", record.clone()));
        assert_eq!(pool.get("coder").unwrap().history.len(), 1);
        assert!(!pool.record_result("nonexistent", record));
    }

    #[test]
    fn test_pool_all() {
        let pool = make_sample_pool();
        let all = pool.all();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_pool_available() {
        let pool = make_sample_pool();
        let available = pool.available();
        assert_eq!(available.len(), 3);

        let mut pool = make_sample_pool();
        pool.get_mut("coder").unwrap().set_load(1.0);
        let available = pool.available();
        assert_eq!(available.len(), 2);
    }

    #[test]
    fn test_pool_online() {
        let pool = make_sample_pool();
        let online = pool.online();
        assert_eq!(online.len(), 3);
    }

    #[test]
    fn test_pool_find_by_skill() {
        let pool = make_sample_pool();
        let coders = pool.find_by_skill(&SkillTag::Coding, 0.5);
        assert_eq!(coders.len(), 1);
        assert_eq!(coders[0].id, "coder");
    }

    #[test]
    fn test_pool_find_qualified() {
        let pool = make_sample_pool();
        let mut required = HashMap::new();
        required.insert(SkillTag::Coding, 0.8);
        required.insert(SkillTag::Testing, 0.5);
        let qualified = pool.find_qualified(&required);
        assert_eq!(qualified.len(), 1);
    }

    #[test]
    fn test_pool_find_qualified_none() {
        let pool = make_sample_pool();
        let mut required = HashMap::new();
        required.insert(SkillTag::Coding, 0.8);
        required.insert(SkillTag::Vision, 0.8); // no single agent has both
        let qualified = pool.find_qualified(&required);
        assert!(qualified.is_empty());
    }

    #[test]
    fn test_pool_stats() {
        let pool = make_sample_pool();
        let stats = pool.stats();
        assert_eq!(stats.total_agents, 3);
        assert_eq!(stats.online_agents, 3);
        assert_eq!(stats.available_agents, 3);
        assert!((stats.avg_load - 0.0).abs() < 0.01);
        assert!(stats.total_available_slots > 0);
        assert!(stats.skill_coverage > 0.0);
    }

    #[test]
    fn test_pool_stats_with_offline() {
        let mut pool = make_sample_pool();
        pool.set_online("coder", false);
        let stats = pool.stats();
        assert_eq!(stats.online_agents, 2);
        assert_eq!(stats.available_agents, 2);
    }

    #[test]
    fn test_pool_events() {
        let mut pool = AgentPool::new();
        let agent = AgentProfile::new("a1", "Alpha", 5);
        pool.register(agent);
        pool.remove("a1");

        let events = pool.events();
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], PoolEvent::AgentRegistered(id) if id == "a1"));
        assert!(matches!(&events[1], PoolEvent::AgentRemoved(id) if id == "a1"));
    }

    #[test]
    fn test_pool_events_trimming() {
        let mut pool = AgentPool::new();
        pool.max_events = 5;

        for i in 0..10 {
            let agent = AgentProfile::new(&format!("a{}", i), &format!("Agent {}", i), 5);
            pool.register(agent);
        }

        // Should have at most max_events
        assert!(pool.events().len() <= pool.max_events);
    }

    #[test]
    fn test_pool_clear_events() {
        let mut pool = make_sample_pool();
        assert!(!pool.events().is_empty());
        pool.clear_events();
        assert!(pool.events().is_empty());
    }

    #[test]
    fn test_pool_default() {
        let pool = AgentPool::default();
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_builder() {
        let pool = AgentPoolBuilder::new()
            .add("a1", "Alpha")
                .capacity(10)
                .skill(SkillTag::Coding, 0.9)
                .done()
            .build();

        assert_eq!(pool.len(), 1);
        let agent = pool.get("a1").unwrap();
        assert_eq!(agent.max_capacity, 10);
        assert!((agent.skill_level(&SkillTag::Coding) - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_pool_builder_offline() {
        let pool = AgentPoolBuilder::new()
            .add("a1", "Alpha")
                .offline()
                .done()
            .build();

        assert!(!pool.get("a1").unwrap().online);
    }
}
