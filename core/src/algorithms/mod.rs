pub mod linear_fa;
pub mod mock;
pub mod tabular_q;
pub mod tiny_nn;

pub use linear_fa::LinearFA;
pub use mock::MockPolicy;
pub use tabular_q::TabularQLearning;
pub use tiny_nn::TinyNN;

pub use crate::Policy;

/// Common utilities for algorithms
pub mod utils {
    use crate::action::Action;
    use crate::obs::Obs;

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
            let sum: f32 = input
                .as_slice()
                .iter()
                .zip(weight_row.iter())
                .map(|(x, w)| x * w)
                .sum();
            output[i] = sum + bias_val;
        }

        Action::new(output)
    }

    /// Matrix-vector multiplication using the crate SIMD backend (scalar fallback when needed).
    #[cfg(feature = "std")]
    pub fn matrix_vector_mul_backend<const IN: usize, const OUT: usize>(
        input: &Obs<IN>,
        weights: &[[f32; IN]; OUT],
        bias: &[f32; OUT],
    ) -> Action<OUT> {
        crate::simd::matrix_vector_mul(input, weights, bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Obs;

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
