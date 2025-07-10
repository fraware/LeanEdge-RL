use crate::error::{Error, Result};

/// Fixed-size observation array
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obs<const N: usize> {
    data: [f32; N],
}

impl<const N: usize> Obs<N> {
    /// Create a new observation from array
    pub fn new(data: [f32; N]) -> Self {
        Self { data }
    }
    
    /// Create observation from slice
    pub fn from_slice(slice: &[f32]) -> Result<Self> {
        if slice.len() != N {
            return Err(Error::InvalidObsSize {
                expected: N,
                actual: slice.len(),
            });
        }
        
        let mut data = [0.0; N];
        data.copy_from_slice(slice);
        Ok(Self { data })
    }
    
    /// Get reference to underlying array
    pub fn as_array(&self) -> &[f32; N] {
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
        if index >= N {
            return Err(Error::InvalidObsSize {
                expected: N,
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
    
    /// Dot product with another observation
    pub fn dot(&self, other: &Self) -> f32 {
        self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum()
    }
    
    /// L2 norm
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
    
    /// Normalize to unit vector
    pub fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm == 0.0 {
            return Self { data: [0.0; N] };
        }
        self.map(|x| x / norm)
    }
}

impl<const N: usize> Default for Obs<N> {
    fn default() -> Self {
        Self { data: [0.0; N] }
    }
}

impl<const N: usize> From<[f32; N]> for Obs<N> {
    fn from(data: [f32; N]) -> Self {
        Self::new(data)
    }
}

impl<const N: usize> TryFrom<&[f32]> for Obs<N> {
    type Error = Error;
    
    fn try_from(slice: &[f32]) -> Result<Self> {
        Self::from_slice(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_obs_creation() {
        let obs = Obs::new([1.0, 2.0, 3.0]);
        assert_eq!(obs.as_slice(), [1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_obs_from_slice() {
        let obs = Obs::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(obs.as_slice(), [1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_obs_from_slice_wrong_size() {
        let result = Obs::from_slice(&[1.0, 2.0]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_obs_operations() {
        let obs1 = Obs::new([1.0, 2.0, 3.0]);
        let obs2 = Obs::new([4.0, 5.0, 6.0]);
        
        let sum = obs1.add(&obs2);
        assert_eq!(sum.as_slice(), [5.0, 7.0, 9.0]);
        
        let dot = obs1.dot(&obs2);
        assert_eq!(dot, 32.0);
    }
    
    #[test]
    fn test_obs_normalize() {
        let obs = Obs::new([3.0, 4.0]);
        let normalized = obs.normalize();
        let norm = normalized.norm();
        assert!((norm - 1.0).abs() < 1e-6);
    }
} 