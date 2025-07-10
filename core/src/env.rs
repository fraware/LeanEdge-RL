use crate::{
    error::{Error, Result},
    obs::Obs,
    action::Action,
    algorithms::{Policy, TabularQLearning, LinearFA, TinyNN},
};

/// Environment state for tracking internal state
#[derive(Debug, Clone, PartialEq)]
pub struct EnvState {
    /// Current observation
    pub current_obs: Vec<f32>,
    /// Step counter
    pub step_count: u64,
    /// Episode counter
    pub episode_count: u64,
    /// Algorithm type
    pub algorithm: AlgorithmType,
    /// Policy weights hash for verification
    pub weights_hash: [u8; 32],
}

/// Supported RL algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlgorithmType {
    TabularQLearning,
    LinearFA,
    TinyNN,
}

impl AlgorithmType {
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::TabularQLearning),
            1 => Ok(Self::LinearFA),
            2 => Ok(Self::TinyNN),
            _ => Err(Error::UnsupportedAlgorithm(format!("Unknown algorithm: {}", value))),
        }
    }
    
    pub fn to_u8(self) -> u8 {
        match self {
            Self::TabularQLearning => 0,
            Self::LinearFA => 1,
            Self::TinyNN => 2,
        }
    }
}

/// Main environment struct implementing the RL interface
pub struct Env<'a, const OBS_DIM: usize, const ACTION_DIM: usize> {
    state: EnvState,
    policy: Box<dyn Policy<OBS_DIM, ACTION_DIM> + 'a>,
}

impl<'a, const OBS_DIM: usize, const ACTION_DIM: usize> Env<'a, OBS_DIM, ACTION_DIM> {
    /// Create environment from weights
    pub fn from_weights(weights: &'a [u8]) -> Result<Self> {
        if weights.len() < 1 {
            return Err(Error::InvalidWeights("Empty weights data".to_string()));
        }
        
        let algorithm = AlgorithmType::from_u8(weights[0])?;
        let policy_weights = &weights[1..];
        
        let policy: Box<dyn Policy<OBS_DIM, ACTION_DIM>> = match algorithm {
            AlgorithmType::TabularQLearning => {
                Box::new(TabularQLearning::from_weights(policy_weights)?)
            }
            AlgorithmType::LinearFA => {
                Box::new(LinearFA::from_weights(policy_weights)?)
            }
            AlgorithmType::TinyNN => {
                Box::new(TinyNN::from_weights(policy_weights)?)
            }
        };
        
        // Calculate weights hash for verification
        let mut weights_hash = [0u8; 32];
        if weights.len() >= 32 {
            weights_hash.copy_from_slice(&weights[..32]);
        }
        
        let state = EnvState {
            current_obs: vec![0.0; OBS_DIM],
            step_count: 0,
            episode_count: 0,
            algorithm,
            weights_hash,
        };
        
        Ok(Self { state, policy })
    }
    
    /// Reset environment with initial observation
    pub fn reset(&mut self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM> {
        self.state.current_obs = obs.as_slice().to_vec();
        self.state.step_count = 0;
        self.state.episode_count += 1;
        
        // Compute initial action
        self.policy.act(obs)
    }
    
    /// Step environment with new observation
    pub fn step(&mut self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM> {
        self.state.current_obs = obs.as_slice().to_vec();
        self.state.step_count += 1;
        
        // Compute action
        self.policy.act(obs)
    }
    
    /// Get current environment state
    pub fn state(&self) -> &EnvState {
        &self.state
    }
    
    /// Set environment state (for testing/debugging)
    pub fn set_state(&mut self, state: EnvState) {
        self.state = state;
    }
    
    /// Update policy weights
    pub fn update_weights(&mut self, weights: &[u8]) -> Result<()> {
        if weights.len() < 1 {
            return Err(Error::InvalidWeights("Empty weights data".to_string()));
        }
        
        let algorithm = AlgorithmType::from_u8(weights[0])?;
        if algorithm != self.state.algorithm {
            return Err(Error::InvalidWeights("Algorithm type mismatch".to_string()));
        }
        
        self.policy.update_weights(weights)?;
        
        // Update weights hash
        if weights.len() >= 32 {
            self.state.weights_hash.copy_from_slice(&weights[..32]);
        }
        
        Ok(())
    }
    
    /// Get policy weights for serialization
    pub fn get_weights(&self) -> Result<Vec<u8>> {
        let mut weights = vec![self.state.algorithm.to_u8()];
        weights.extend(self.policy.get_weights()?);
        Ok(weights)
    }
    
    /// Verify safety invariant
    pub fn check_invariant(&self, obs: &Obs<OBS_DIM>, action: &Action<ACTION_DIM>) -> Result<()> {
        // Basic safety checks
        if !action.is_within_bounds(-1.0, 1.0) {
            return Err(Error::InvariantViolation("Action out of bounds".to_string()));
        }
        
        // Check for NaN or infinite values
        if obs.as_slice().iter().any(|x| !x.is_finite()) {
            return Err(Error::InvariantViolation("Observation contains NaN or infinite values".to_string()));
        }
        
        if action.as_slice().iter().any(|x| !x.is_finite()) {
            return Err(Error::InvariantViolation("Action contains NaN or infinite values".to_string()));
        }
        
        Ok(())
    }
}

impl<'a, const OBS_DIM: usize, const ACTION_DIM: usize> crate::Environment<OBS_DIM, ACTION_DIM> 
    for Env<'a, OBS_DIM, ACTION_DIM> 
{
    fn reset(&mut self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM> {
        self.reset(obs)
    }
    
    fn step(&mut self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM> {
        self.step(obs)
    }
    
    fn state(&self) -> &EnvState {
        self.state()
    }
    
    fn set_state(&mut self, state: EnvState) {
        self.set_state(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::mock::MockPolicy;
    
    #[test]
    fn test_env_creation() {
        let mut weights = vec![0u8]; // TabularQLearning
        weights.extend(vec![1.0f32.to_le_bytes().to_vec()].concat());
        
        let env = Env::<4, 2>::from_weights(&weights);
        assert!(env.is_ok());
    }
    
    #[test]
    fn test_env_reset_and_step() {
        let mut weights = vec![0u8]; // TabularQLearning
        weights.extend(vec![1.0f32.to_le_bytes().to_vec()].concat());
        
        let mut env = Env::<4, 2>::from_weights(&weights).unwrap();
        let obs = Obs::new([1.0, 2.0, 3.0, 4.0]);
        
        let action = env.reset(&obs);
        assert_eq!(env.state().episode_count, 1);
        assert_eq!(env.state().step_count, 0);
        
        let next_action = env.step(&obs);
        assert_eq!(env.state().step_count, 1);
    }
    
    #[test]
    fn test_env_invariant_check() {
        let mut weights = vec![0u8]; // TabularQLearning
        weights.extend(vec![1.0f32.to_le_bytes().to_vec()].concat());
        
        let env = Env::<4, 2>::from_weights(&weights).unwrap();
        let obs = Obs::new([1.0, 2.0, 3.0, 4.0]);
        let action = Action::new([0.5, -0.3]);
        
        assert!(env.check_invariant(&obs, &action).is_ok());
        
        let bad_action = Action::new([1.5, -0.3]);
        assert!(env.check_invariant(&obs, &bad_action).is_err());
    }
} 