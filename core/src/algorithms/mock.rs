use crate::{
    error::{Error, Result},
    obs::Obs,
    action::Action,
    algorithms::Policy,
};

/// Mock policy for testing purposes
pub struct MockPolicy<const OBS_DIM: usize, const ACTION_DIM: usize> {
    /// Fixed action to return
    fixed_action: Action<ACTION_DIM>,
}

impl<const OBS_DIM: usize, const ACTION_DIM: usize> MockPolicy<OBS_DIM, ACTION_DIM> {
    /// Create new mock policy with fixed action
    pub fn new(fixed_action: Action<ACTION_DIM>) -> Self {
        Self { fixed_action }
    }
    
    /// Create mock policy that returns zeros
    pub fn zeros() -> Self {
        Self {
            fixed_action: Action::new([0.0; ACTION_DIM]),
        }
    }
    
    /// Create mock policy that returns random-like values
    pub fn random_like() -> Self {
        let mut values = [0.0; ACTION_DIM];
        for i in 0..ACTION_DIM {
            values[i] = (i as f32) * 0.1;
        }
        Self {
            fixed_action: Action::new(values),
        }
    }
}

impl<const OBS_DIM: usize, const ACTION_DIM: usize> Policy<OBS_DIM, ACTION_DIM> 
    for MockPolicy<OBS_DIM, ACTION_DIM> 
{
    fn act(&self, _obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM> {
        self.fixed_action
    }
    
    fn update_weights(&mut self, _weights: &[u8]) -> Result<()> {
        // Mock policy doesn't use weights
        Ok(())
    }
    
    fn get_weights(&self) -> Result<Vec<u8>> {
        // Return empty weights for mock policy
        Ok(Vec::new())
    }
    
    fn algorithm_name(&self) -> &'static str {
        "MockPolicy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_policy_creation() {
        let action = Action::new([0.5, -0.3]);
        let policy = MockPolicy::<4, 2>::new(action);
        
        let obs = Obs::new([1.0, 2.0, 3.0, 4.0]);
        let result = policy.act(&obs);
        
        assert_eq!(result.as_slice(), [0.5, -0.3]);
    }
    
    #[test]
    fn test_mock_policy_zeros() {
        let policy = MockPolicy::<4, 2>::zeros();
        let obs = Obs::new([1.0, 2.0, 3.0, 4.0]);
        let result = policy.act(&obs);
        
        assert_eq!(result.as_slice(), [0.0, 0.0]);
    }
    
    #[test]
    fn test_mock_policy_random_like() {
        let policy = MockPolicy::<4, 2>::random_like();
        let obs = Obs::new([1.0, 2.0, 3.0, 4.0]);
        let result = policy.act(&obs);
        
        assert_eq!(result.as_slice(), [0.0, 0.1]);
    }
} 