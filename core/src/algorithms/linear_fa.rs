use crate::{
    error::{Error, Result},
    obs::Obs,
    action::Action,
    algorithms::{Policy, utils},
};

/// Linear Function Approximation implementation
pub struct LinearFA<const OBS_DIM: usize, const ACTION_DIM: usize> {
    /// Weight matrix: [action_dim][obs_dim]
    weights: Vec<Vec<f32>>,
    /// Bias vector: [action_dim]
    bias: Vec<f32>,
    /// Learning rate
    alpha: f32,
}

impl<const OBS_DIM: usize, const ACTION_DIM: usize> LinearFA<OBS_DIM, ACTION_DIM> {
    /// Create new LinearFA with random weights
    pub fn new() -> Self {
        let mut weights = vec![vec![0.0; OBS_DIM]; ACTION_DIM];
        let bias = vec![0.0; ACTION_DIM];
        
        // Initialize with small random weights
        for i in 0..ACTION_DIM {
            for j in 0..OBS_DIM {
                weights[i][j] = (i as f32 + j as f32) * 0.01;
            }
        }
        
        Self {
            weights,
            bias,
            alpha: 0.01,
        }
    }
    
    /// Create from weights
    pub fn from_weights(weights: &[u8]) -> Result<Self> {
        let header_size = 4; // alpha (f32)
        if weights.len() < header_size {
            return Err(Error::InvalidWeights("Insufficient weights for LinearFA".to_string()));
        }
        
        let alpha = f32::from_le_bytes([weights[0], weights[1], weights[2], weights[3]]);
        
        let expected_size = header_size + (OBS_DIM * ACTION_DIM + ACTION_DIM) * 4; // 4 bytes per f32
        if weights.len() < expected_size {
            return Err(Error::InvalidWeights("Insufficient weights for LinearFA".to_string()));
        }
        
        let mut lfa = Self::new();
        lfa.alpha = alpha;
        
        // Load weights matrix
        let weights_data = &weights[header_size..header_size + OBS_DIM * ACTION_DIM * 4];
        lfa.load_weights_matrix(weights_data)?;
        
        // Load bias vector
        let bias_data = &weights[header_size + OBS_DIM * ACTION_DIM * 4..expected_size];
        lfa.load_bias_vector(bias_data)?;
        
        Ok(lfa)
    }
    
    /// Load weights matrix from bytes
    fn load_weights_matrix(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != OBS_DIM * ACTION_DIM * 4 {
            return Err(Error::InvalidWeights("Weights matrix size mismatch".to_string()));
        }
        
        for (i, chunk) in data.chunks(4).enumerate() {
            let action_idx = i / OBS_DIM;
            let obs_idx = i % OBS_DIM;
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            self.weights[action_idx][obs_idx] = value;
        }
        
        Ok(())
    }
    
    /// Load bias vector from bytes
    fn load_bias_vector(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != ACTION_DIM * 4 {
            return Err(Error::InvalidWeights("Bias vector size mismatch".to_string()));
        }
        
        for (i, chunk) in data.chunks(4).enumerate() {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            self.bias[i] = value;
        }
        
        Ok(())
    }
    
    /// Compute linear transformation: action = weights * obs + bias
    fn compute_action(&self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM> {
        let mut action_values = [0.0; ACTION_DIM];
        
        for (action_idx, (weight_row, &bias_val)) in self.weights.iter().zip(self.bias.iter()).enumerate() {
            let sum: f32 = obs.as_slice().iter().zip(weight_row.iter()).map(|(x, w)| x * w).sum();
            action_values[action_idx] = sum + bias_val;
        }
        
        Action::new(action_values)
    }
    
    /// Update weights using gradient descent
    pub fn update_weights(&mut self, obs: &Obs<OBS_DIM>, target_action: &Action<ACTION_DIM>, current_action: &Action<ACTION_DIM>) {
        let error = target_action.sub(current_action);
        
        for (action_idx, error_val) in error.as_slice().iter().enumerate() {
            let gradient = *error_val * self.alpha;
            
            // Update weights
            for (obs_idx, obs_val) in obs.as_slice().iter().enumerate() {
                self.weights[action_idx][obs_idx] += gradient * obs_val;
            }
            
            // Update bias
            self.bias[action_idx] += gradient;
        }
    }
    
    /// Set learning rate
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;
    }
    
    /// Get weight at specific position
    pub fn get_weight(&self, action_idx: usize, obs_idx: usize) -> f32 {
        self.weights[action_idx][obs_idx]
    }
    
    /// Get bias for specific action
    pub fn get_bias(&self, action_idx: usize) -> f32 {
        self.bias[action_idx]
    }
}

impl<const OBS_DIM: usize, const ACTION_DIM: usize> Policy<OBS_DIM, ACTION_DIM> 
    for LinearFA<OBS_DIM, ACTION_DIM> 
{
    fn act(&self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM> {
        let action = self.compute_action(obs);
        
        // Apply tanh activation to bound actions to [-1, 1]
        action.map(|x| utils::tanh(x))
    }
    
    fn update_weights(&mut self, weights: &[u8]) -> Result<()> {
        let header_size = 4; // alpha (f32)
        if weights.len() < header_size {
            return Err(Error::InvalidWeights("Insufficient weights for LinearFA".to_string()));
        }
        
        let alpha = f32::from_le_bytes([weights[0], weights[1], weights[2], weights[3]]);
        self.alpha = alpha;
        
        let expected_size = header_size + (OBS_DIM * ACTION_DIM + ACTION_DIM) * 4;
        if weights.len() >= expected_size {
            let weights_data = &weights[header_size..header_size + OBS_DIM * ACTION_DIM * 4];
            self.load_weights_matrix(weights_data)?;
            
            let bias_data = &weights[header_size + OBS_DIM * ACTION_DIM * 4..expected_size];
            self.load_bias_vector(bias_data)?;
        }
        
        Ok(())
    }
    
    fn get_weights(&self) -> Result<Vec<u8>> {
        let mut weights = Vec::new();
        
        // Header: alpha
        weights.extend(self.alpha.to_le_bytes());
        
        // Weights matrix
        for action_idx in 0..ACTION_DIM {
            for obs_idx in 0..OBS_DIM {
                weights.extend(self.weights[action_idx][obs_idx].to_le_bytes());
            }
        }
        
        // Bias vector
        for &bias_val in &self.bias {
            weights.extend(bias_val.to_le_bytes());
        }
        
        Ok(weights)
    }
    
    fn algorithm_name(&self) -> &'static str {
        "LinearFA"
    }
}

impl<const OBS_DIM: usize, const ACTION_DIM: usize> Default for LinearFA<OBS_DIM, ACTION_DIM> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_fa_creation() {
        let lfa = LinearFA::<4, 2>::new();
        assert_eq!(lfa.weights.len(), 2);
        assert_eq!(lfa.weights[0].len(), 4);
        assert_eq!(lfa.bias.len(), 2);
    }
    
    #[test]
    fn test_linear_fa_from_weights() {
        let mut weights = Vec::new();
        weights.extend((0.01f32).to_le_bytes()); // alpha
        
        // Add weights matrix (4*2 = 8 floats)
        for i in 0..8 {
            weights.extend((i as f32 * 0.1).to_le_bytes());
        }
        
        // Add bias vector (2 floats)
        weights.extend((0.1f32).to_le_bytes());
        weights.extend((0.2f32).to_le_bytes());
        
        let lfa = LinearFA::<4, 2>::from_weights(&weights);
        assert!(lfa.is_ok());
        
        let lfa = lfa.unwrap();
        assert_eq!(lfa.alpha, 0.01);
        assert_eq!(lfa.weights[0][0], 0.0);
        assert_eq!(lfa.weights[0][1], 0.1);
        assert_eq!(lfa.bias[0], 0.1);
        assert_eq!(lfa.bias[1], 0.2);
    }
    
    #[test]
    fn test_linear_fa_action() {
        let lfa = LinearFA::<4, 2>::new();
        let obs = Obs::new([1.0, 2.0, 3.0, 4.0]);
        let action = lfa.act(&obs);
        
        // Should return bounded actions
        assert_eq!(action.as_slice().len(), 2);
        for &val in action.as_slice() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }
    
    #[test]
    fn test_linear_fa_weight_update() {
        let mut lfa = LinearFA::<4, 2>::new();
        let obs = Obs::new([1.0, 2.0, 3.0, 4.0]);
        let current_action = lfa.act(&obs);
        let target_action = Action::new([0.5, -0.3]);
        
        let old_weight = lfa.get_weight(0, 0);
        lfa.update_weights(&obs, &target_action, &current_action);
        let new_weight = lfa.get_weight(0, 0);
        
        // Weight should have changed
        assert_ne!(old_weight, new_weight);
    }
} 