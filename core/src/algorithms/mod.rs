pub mod tabular_q;
pub mod linear_fa;
pub mod tiny_nn;
pub mod mock;

pub use tabular_q::TabularQLearning;
pub use linear_fa::LinearFA;
pub use tiny_nn::TinyNN;
pub use mock::MockPolicy;

use crate::{
    error::{Error, Result},
    obs::Obs,
    action::Action,
};

/// Policy trait for different RL algorithms
pub trait Policy<const OBS_DIM: usize, const ACTION_DIM: usize> {
    /// Compute action from observation
    fn act(&self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM>;
    
    /// Update policy weights
    fn update_weights(&mut self, weights: &[u8]) -> Result<()>;
    
    /// Get policy weights for serialization
    fn get_weights(&self) -> Result<Vec<u8>>;
    
    /// Get algorithm name
    fn algorithm_name(&self) -> &'static str;
}

/// Common utilities for algorithms
pub mod utils {
    use crate::obs::Obs;
    use crate::action::Action;
    
    /// Apply ReLU activation function
    pub fn relu(x: f32) -> f32 {
        x.max(0.0)
    }
    
    /// Apply tanh activation function
    pub fn tanh(x: f32) -> f32 {
        x.tanh()
    }
    
    /// Apply sigmoid activation function
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Linear transformation: y = Wx + b
    pub fn linear_transform<const IN: usize, const OUT: usize>(
        input: &Obs<IN>,
        weights: &[[f32; IN]; OUT],
        bias: &[f32; OUT],
    ) -> Action<OUT> {
        let mut output = [0.0; OUT];
        
        for (i, (weight_row, &bias_val)) in weights.iter().zip(bias.iter()).enumerate() {
            let sum: f32 = input.as_slice().iter().zip(weight_row.iter()).map(|(x, w)| x * w).sum();
            output[i] = sum + bias_val;
        }
        
        Action::new(output)
    }
    
    /// Matrix-vector multiplication with SIMD optimization
    #[cfg(feature = "simd_avx2")]
    pub fn matrix_vector_mul_simd<const IN: usize, const OUT: usize>(
        input: &Obs<IN>,
        weights: &[[f32; IN]; OUT],
        bias: &[f32; OUT],
    ) -> Action<OUT> {
        use std::arch::x86_64::*;
        
        let mut output = [0.0; OUT];
        
        for (i, (weight_row, &bias_val)) in weights.iter().zip(bias.iter()).enumerate() {
            let mut sum = _mm256_setzero_ps();
            let mut j = 0;
            
            // Process 8 elements at a time with AVX2
            while j + 8 <= IN {
                let input_vec = _mm256_loadu_ps(&input.as_slice()[j]);
                let weight_vec = _mm256_loadu_ps(&weight_row[j]);
                sum = _mm256_fmadd_ps(input_vec, weight_vec, sum);
                j += 8;
            }
            
            // Handle remaining elements
            let mut scalar_sum = 0.0;
            for k in j..IN {
                scalar_sum += input.as_slice()[k] * weight_row[k];
            }
            
            // Reduce SIMD sum
            let simd_sum = unsafe {
                let mut temp = [0.0f32; 8];
                _mm256_storeu_ps(temp.as_mut_ptr(), sum);
                temp.iter().sum::<f32>()
            };
            
            output[i] = simd_sum + scalar_sum + bias_val;
        }
        
        Action::new(output)
    }
    
    /// NEON-optimized matrix-vector multiplication
    #[cfg(feature = "simd_neon")]
    pub fn matrix_vector_mul_neon<const IN: usize, const OUT: usize>(
        input: &Obs<IN>,
        weights: &[[f32; IN]; OUT],
        bias: &[f32; OUT],
    ) -> Action<OUT> {
        use std::arch::aarch64::*;
        
        let mut output = [0.0; OUT];
        
        for (i, (weight_row, &bias_val)) in weights.iter().zip(bias.iter()).enumerate() {
            let mut sum = unsafe { vdupq_n_f32(0.0) };
            let mut j = 0;
            
            // Process 4 elements at a time with NEON
            while j + 4 <= IN {
                let input_vec = unsafe { vld1q_f32(&input.as_slice()[j]) };
                let weight_vec = unsafe { vld1q_f32(&weight_row[j]) };
                sum = unsafe { vmlaq_f32(sum, input_vec, weight_vec) };
                j += 4;
            }
            
            // Handle remaining elements
            let mut scalar_sum = 0.0;
            for k in j..IN {
                scalar_sum += input.as_slice()[k] * weight_row[k];
            }
            
            // Reduce NEON sum
            let simd_sum = unsafe {
                let mut temp = [0.0f32; 4];
                vst1q_f32(temp.as_mut_ptr(), sum);
                temp.iter().sum::<f32>()
            };
            
            output[i] = simd_sum + scalar_sum + bias_val;
        }
        
        Action::new(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_transform() {
        let input = Obs::new([1.0, 2.0]);
        let weights = [[1.0, 0.5], [0.5, 1.0]];
        let bias = [0.1, 0.2];
        
        let output = utils::linear_transform(&input, &weights, &bias);
        assert_eq!(output.as_slice(), [2.1, 2.7]); // [1*1 + 2*0.5 + 0.1, 1*0.5 + 2*1 + 0.2]
    }
    
    #[test]
    fn test_activation_functions() {
        assert_eq!(utils::relu(1.0), 1.0);
        assert_eq!(utils::relu(-1.0), 0.0);
        assert_eq!(utils::relu(0.0), 0.0);
        
        assert!((utils::tanh(0.0)).abs() < 1e-6);
        assert!((utils::sigmoid(0.0) - 0.5).abs() < 1e-6);
    }
} 