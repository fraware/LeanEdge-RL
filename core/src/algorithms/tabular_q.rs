use crate::{
    error::{Error, Result},
    obs::Obs,
    action::Action,
    algorithms::{Policy, utils},
};

/// Tabular Q-Learning implementation
pub struct TabularQLearning<const OBS_DIM: usize, const ACTION_DIM: usize> {
    /// Q-table: [state][action] -> Q-value
    q_table: Vec<Vec<f32>>,
    /// Learning rate
    alpha: f32,
    /// Discount factor
    gamma: f32,
    /// Epsilon for epsilon-greedy exploration
    epsilon: f32,
    /// Number of discrete states
    num_states: usize,
    /// Number of discrete actions
    num_actions: usize,
}

impl<const OBS_DIM: usize, const ACTION_DIM: usize> TabularQLearning<OBS_DIM, ACTION_DIM> {
    /// Create new TabularQLearning with default parameters
    pub fn new(num_states: usize, num_actions: usize) -> Self {
        Self {
            q_table: vec![vec![0.0; num_actions]; num_states],
            alpha: 0.1,
            gamma: 0.9,
            epsilon: 0.1,
            num_states,
            num_actions,
        }
    }
    
    /// Create from weights
    pub fn from_weights(weights: &[u8]) -> Result<Self> {
        if weights.len() < 16 {
            return Err(Error::InvalidWeights("Insufficient weights for TabularQLearning".to_string()));
        }
        
        // Parse header: [num_states, num_actions, alpha, gamma, epsilon] (4 bytes each)
        let num_states = u32::from_le_bytes([weights[0], weights[1], weights[2], weights[3]]) as usize;
        let num_actions = u32::from_le_bytes([weights[4], weights[5], weights[6], weights[7]]) as usize;
        let alpha = f32::from_le_bytes([weights[8], weights[9], weights[10], weights[11]]);
        let gamma = f32::from_le_bytes([weights[12], weights[13], weights[14], weights[15]]);
        
        let mut ql = Self::new(num_states, num_actions);
        ql.alpha = alpha;
        ql.gamma = gamma;
        
        // Load Q-table if provided
        let expected_q_table_size = num_states * num_actions * 4; // 4 bytes per f32
        if weights.len() >= 16 + expected_q_table_size {
            let q_table_data = &weights[16..16 + expected_q_table_size];
            ql.load_q_table(q_table_data)?;
        }
        
        Ok(ql)
    }
    
    /// Load Q-table from bytes
    fn load_q_table(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != self.num_states * self.num_actions * 4 {
            return Err(Error::InvalidWeights("Q-table size mismatch".to_string()));
        }
        
        for (i, chunk) in data.chunks(4).enumerate() {
            let state = i / self.num_actions;
            let action = i % self.num_actions;
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            self.q_table[state][action] = value;
        }
        
        Ok(())
    }
    
    /// Discretize continuous observation to state index
    fn discretize_obs(&self, obs: &Obs<OBS_DIM>) -> usize {
        // Simple discretization: use first observation value as state
        // In practice, this would be more sophisticated
        let value = obs.as_slice()[0];
        let state = ((value + 1.0) * (self.num_states as f32) / 2.0) as usize;
        state.clamp(0, self.num_states - 1)
    }
    
    /// Epsilon-greedy action selection
    fn select_action(&self, state: usize) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // Use current time as seed for exploration
        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now().hash(&mut hasher);
        let random_value = (hasher.finish() % 1000) as f32 / 1000.0;
        
        if random_value < self.epsilon {
            // Explore: random action
            (hasher.finish() % self.num_actions as u64) as usize
        } else {
            // Exploit: best action
            self.q_table[state]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(action, _)| action)
                .unwrap_or(0)
        }
    }
    
    /// Update Q-value using Q-learning update rule
    pub fn update_q_value(&mut self, state: usize, action: usize, reward: f32, next_state: usize) {
        let current_q = self.q_table[state][action];
        let max_next_q = self.q_table[next_state].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q);
        self.q_table[state][action] = new_q;
    }
    
    /// Set epsilon for exploration
    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon.clamp(0.0, 1.0);
    }
    
    /// Get Q-value for state-action pair
    pub fn get_q_value(&self, state: usize, action: usize) -> f32 {
        self.q_table[state][action]
    }
}

impl<const OBS_DIM: usize, const ACTION_DIM: usize> Policy<OBS_DIM, ACTION_DIM> 
    for TabularQLearning<OBS_DIM, ACTION_DIM> 
{
    fn act(&self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM> {
        let state = self.discretize_obs(obs);
        let action_idx = self.select_action(state);
        
        // Convert discrete action to continuous action
        let mut action_values = [0.0; ACTION_DIM];
        if action_idx < ACTION_DIM {
            action_values[action_idx] = 1.0;
        }
        
        Action::new(action_values)
    }
    
    fn update_weights(&mut self, weights: &[u8]) -> Result<()> {
        if weights.len() < 16 {
            return Err(Error::InvalidWeights("Insufficient weights for TabularQLearning".to_string()));
        }
        
        let num_states = u32::from_le_bytes([weights[0], weights[1], weights[2], weights[3]]) as usize;
        let num_actions = u32::from_le_bytes([weights[4], weights[5], weights[6], weights[7]]) as usize;
        
        if num_states != self.num_states || num_actions != self.num_actions {
            return Err(Error::InvalidWeights("State/action dimensions mismatch".to_string()));
        }
        
        let alpha = f32::from_le_bytes([weights[8], weights[9], weights[10], weights[11]]);
        let gamma = f32::from_le_bytes([weights[12], weights[13], weights[14], weights[15]]);
        
        self.alpha = alpha;
        self.gamma = gamma;
        
        // Update Q-table if provided
        let expected_q_table_size = num_states * num_actions * 4;
        if weights.len() >= 16 + expected_q_table_size {
            let q_table_data = &weights[16..16 + expected_q_table_size];
            self.load_q_table(q_table_data)?;
        }
        
        Ok(())
    }
    
    fn get_weights(&self) -> Result<Vec<u8>> {
        let mut weights = Vec::new();
        
        // Header: [num_states, num_actions, alpha, gamma, epsilon]
        weights.extend((self.num_states as u32).to_le_bytes());
        weights.extend((self.num_actions as u32).to_le_bytes());
        weights.extend(self.alpha.to_le_bytes());
        weights.extend(self.gamma.to_le_bytes());
        
        // Q-table
        for state in &self.q_table {
            for &q_value in state {
                weights.extend(q_value.to_le_bytes());
            }
        }
        
        Ok(weights)
    }
    
    fn algorithm_name(&self) -> &'static str {
        "TabularQLearning"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tabular_q_creation() {
        let ql = TabularQLearning::<4, 2>::new(10, 3);
        assert_eq!(ql.num_states, 10);
        assert_eq!(ql.num_actions, 3);
        assert_eq!(ql.q_table.len(), 10);
        assert_eq!(ql.q_table[0].len(), 3);
    }
    
    #[test]
    fn test_tabular_q_from_weights() {
        let mut weights = Vec::new();
        weights.extend((5u32).to_le_bytes()); // num_states
        weights.extend((2u32).to_le_bytes()); // num_actions
        weights.extend((0.1f32).to_le_bytes()); // alpha
        weights.extend((0.9f32).to_le_bytes()); // gamma
        
        let ql = TabularQLearning::<4, 2>::from_weights(&weights);
        assert!(ql.is_ok());
        
        let ql = ql.unwrap();
        assert_eq!(ql.num_states, 5);
        assert_eq!(ql.num_actions, 2);
        assert_eq!(ql.alpha, 0.1);
        assert_eq!(ql.gamma, 0.9);
    }
    
    #[test]
    fn test_tabular_q_action() {
        let ql = TabularQLearning::<4, 2>::new(10, 2);
        let obs = Obs::new([0.5, 0.0, 0.0, 0.0]);
        let action = ql.act(&obs);
        
        // Should return a valid action
        assert_eq!(action.as_slice().len(), 2);
        assert!(action.as_slice().iter().any(|&x| x > 0.0));
    }
    
    #[test]
    fn test_tabular_q_update() {
        let mut ql = TabularQLearning::<4, 2>::new(10, 2);
        let state = 0;
        let action = 1;
        let reward = 1.0;
        let next_state = 1;
        
        let old_q = ql.get_q_value(state, action);
        ql.update_q_value(state, action, reward, next_state);
        let new_q = ql.get_q_value(state, action);
        
        assert!(new_q > old_q); // Q-value should increase with positive reward
    }
} 