use crate::error::{Error, Result};

/// Fixed-size action array
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Action<const M: usize> {
    data: [f32; M],
}

impl<const M: usize> Action<M> {
    /// Create a new action from array
    pub fn new(data: [f32; M]) -> Self {
        Self { data }
    }
    
    /// Create action from slice
    pub fn from_slice(slice: &[f32]) -> Result<Self> {
        if slice.len() != M {
            return Err(Error::InvalidActionSize {
                expected: M,
                actual: slice.len(),
            });
        }
        
        let mut data = [0.0; M];
        data.copy_from_slice(slice);
        Ok(Self { data })
    }
    
    /// Get reference to underlying array
    pub fn as_array(&self) -> &[f32; M] {
        &self.data
    }
    
    /// Get slice of the data
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
    
    /// Get mutable slice of the data
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
    
    /// Get element at index
    pub fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).copied()
    }
    
    /// Set element at index
    pub fn set(&mut self, index: usize, value: f32) -> Result<()> {
        if index >= M {
            return Err(Error::InvalidActionSize {
                expected: M,
                actual: index + 1,
            });
        }
        self.data[index] = value;
        Ok(())
    }
    
    /// Apply function to each element
    pub fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(f32) -> f32,
    {
        let data = self.data.map(f);
        Self { data }
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        let data = self.data.zip(other.data).map(|(a, b)| a + b);
        Self { data }
    }
    
    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Self {
        let data = self.data.zip(other.data).map(|(a, b)| a - b);
        Self { data }
    }
    
    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> Self {
        let data = self.data.zip(other.data).map(|(a, b)| a * b);
        Self { data }
    }
    
    /// Element-wise division
    pub fn div(&self, other: &Self) -> Result<Self> {
        let data = self.data.zip(other.data).map(|(a, b)| {
            if b == 0.0 {
                return f32::INFINITY; // Handle division by zero
            }
            a / b
        });
        Ok(Self { data })
    }
    
    /// Scale action by scalar
    pub fn scale(&self, factor: f32) -> Self {
        self.map(|x| x * factor)
    }
    
    /// Clamp action values to range
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        self.map(|x| x.clamp(min, max))
    }
    
    /// Apply softmax to action values
    pub fn softmax(&self) -> Self {
        let max_val = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = self.data.iter().map(|x| (x - max_val).exp()).sum();
        let data = self.data.map(|x| (x - max_val).exp() / exp_sum);
        Self { data }
    }
    
    /// Get index of maximum value
    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0)
    }
    
    /// Get maximum value
    pub fn max(&self) -> f32 {
        self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }
    
    /// Get minimum value
    pub fn min(&self) -> f32 {
        self.data.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }
    
    /// Check if action is within bounds
    pub fn is_within_bounds(&self, min: f32, max: f32) -> bool {
        self.data.iter().all(|&x| x >= min && x <= max)
    }
}

impl<const M: usize> Default for Action<M> {
    fn default() -> Self {
        Self { data: [0.0; M] }
    }
}

impl<const M: usize> From<[f32; M]> for Action<M> {
    fn from(data: [f32; M]) -> Self {
        Self::new(data)
    }
}

impl<const M: usize> TryFrom<&[f32]> for Action<M> {
    type Error = Error;
    
    fn try_from(slice: &[f32]) -> Result<Self> {
        Self::from_slice(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_action_creation() {
        let action = Action::new([0.5, -0.3]);
        assert_eq!(action.as_slice(), [0.5, -0.3]);
    }
    
    #[test]
    fn test_action_from_slice() {
        let action = Action::from_slice(&[0.5, -0.3]).unwrap();
        assert_eq!(action.as_slice(), [0.5, -0.3]);
    }
    
    #[test]
    fn test_action_from_slice_wrong_size() {
        let result = Action::from_slice(&[0.5]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_action_operations() {
        let action1 = Action::new([0.5, -0.3]);
        let action2 = Action::new([0.2, 0.1]);
        
        let sum = action1.add(&action2);
        assert_eq!(sum.as_slice(), [0.7, -0.2]);
        
        let scaled = action1.scale(2.0);
        assert_eq!(scaled.as_slice(), [1.0, -0.6]);
    }
    
    #[test]
    fn test_action_clamp() {
        let action = Action::new([1.5, -2.0, 0.5]);
        let clamped = action.clamp(-1.0, 1.0);
        assert_eq!(clamped.as_slice(), [1.0, -1.0, 0.5]);
    }
    
    #[test]
    fn test_action_softmax() {
        let action = Action::new([1.0, 2.0, 3.0]);
        let softmax = action.softmax();
        let sum: f32 = softmax.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_action_argmax() {
        let action = Action::new([0.1, 0.8, 0.3]);
        assert_eq!(action.argmax(), 1);
    }
} 