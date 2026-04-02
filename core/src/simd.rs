// SIMD acceleration module
// Provides optimized implementations for NEON and AVX backends

use crate::{action::Action, obs::Obs};

/// SIMD backend trait (`&self` keeps the trait object-safe for `dyn SimdBackend`).
pub trait SimdBackend {
    /// Matrix-vector multiplication with SIMD acceleration
    fn matrix_vector_mul<const IN: usize, const OUT: usize>(
        &self,
        input: &Obs<IN>,
        weights: &[[f32; IN]; OUT],
        bias: &[f32; OUT],
    ) -> Action<OUT>;

    /// Element-wise vector operations
    fn vector_add<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N];
    fn vector_sub<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N];
    fn vector_mul<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N];
    fn vector_scale<const N: usize>(&self, a: &[f32; N], scale: f32) -> [f32; N];
}

/// Scalar fallback implementation
pub struct ScalarBackend;

impl SimdBackend for ScalarBackend {
    fn matrix_vector_mul<const IN: usize, const OUT: usize>(
        &self,
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

    fn vector_add<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = a[i] + b[i];
        }
        result
    }

    fn vector_sub<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = a[i] - b[i];
        }
        result
    }

    fn vector_mul<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = a[i] * b[i];
        }
        result
    }

    fn vector_scale<const N: usize>(&self, a: &[f32; N], scale: f32) -> [f32; N] {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = a[i] * scale;
        }
        result
    }
}

/// AVX2 backend implementation (x86_64 only; feature-gated for cross-compilation sanity).
#[cfg(all(feature = "simd_avx2", target_arch = "x86_64"))]
pub struct Avx2Backend;

#[cfg(all(feature = "simd_avx2", target_arch = "x86_64"))]
impl SimdBackend for Avx2Backend {
    fn matrix_vector_mul<const IN: usize, const OUT: usize>(
        &self,
        input: &Obs<IN>,
        weights: &[[f32; IN]; OUT],
        bias: &[f32; OUT],
    ) -> Action<OUT> {
        use std::arch::x86_64::*;

        let mut output = [0.0; OUT];

        for (i, (weight_row, &bias_val)) in weights.iter().zip(bias.iter()).enumerate() {
            let mut sum = unsafe { _mm256_setzero_ps() };
            let mut j = 0;

            // Process 8 elements at a time with AVX2
            while j + 8 <= IN {
                let input_vec = unsafe { _mm256_loadu_ps(&input.as_slice()[j]) };
                let weight_vec = unsafe { _mm256_loadu_ps(&weight_row[j]) };
                sum = unsafe { _mm256_fmadd_ps(input_vec, weight_vec, sum) };
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

    fn vector_add<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        use std::arch::x86_64::*;

        let mut result = [0.0; N];
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= N {
            let a_vec = unsafe { _mm256_loadu_ps(&a[i]) };
            let b_vec = unsafe { _mm256_loadu_ps(&b[i]) };
            let sum_vec = unsafe { _mm256_add_ps(a_vec, b_vec) };
            unsafe { _mm256_storeu_ps(&mut result[i], sum_vec) };
            i += 8;
        }

        // Handle remaining elements
        for j in i..N {
            result[j] = a[j] + b[j];
        }

        result
    }

    fn vector_sub<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        use std::arch::x86_64::*;

        let mut result = [0.0; N];
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= N {
            let a_vec = unsafe { _mm256_loadu_ps(&a[i]) };
            let b_vec = unsafe { _mm256_loadu_ps(&b[i]) };
            let diff_vec = unsafe { _mm256_sub_ps(a_vec, b_vec) };
            unsafe { _mm256_storeu_ps(&mut result[i], diff_vec) };
            i += 8;
        }

        // Handle remaining elements
        for j in i..N {
            result[j] = a[j] - b[j];
        }

        result
    }

    fn vector_mul<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        use std::arch::x86_64::*;

        let mut result = [0.0; N];
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= N {
            let a_vec = unsafe { _mm256_loadu_ps(&a[i]) };
            let b_vec = unsafe { _mm256_loadu_ps(&b[i]) };
            let prod_vec = unsafe { _mm256_mul_ps(a_vec, b_vec) };
            unsafe { _mm256_storeu_ps(&mut result[i], prod_vec) };
            i += 8;
        }

        // Handle remaining elements
        for j in i..N {
            result[j] = a[j] * b[j];
        }

        result
    }

    fn vector_scale<const N: usize>(&self, a: &[f32; N], scale: f32) -> [f32; N] {
        use std::arch::x86_64::*;

        let mut result = [0.0; N];
        let scale_vec = unsafe { _mm256_set1_ps(scale) };
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= N {
            let a_vec = unsafe { _mm256_loadu_ps(&a[i]) };
            let scaled_vec = unsafe { _mm256_mul_ps(a_vec, scale_vec) };
            unsafe { _mm256_storeu_ps(&mut result[i], scaled_vec) };
            i += 8;
        }

        // Handle remaining elements
        for j in i..N {
            result[j] = a[j] * scale;
        }

        result
    }
}

/// NEON backend implementation (aarch64 only).
#[cfg(all(feature = "simd_neon", target_arch = "aarch64"))]
pub struct NeonBackend;

#[cfg(all(feature = "simd_neon", target_arch = "aarch64"))]
impl SimdBackend for NeonBackend {
    fn matrix_vector_mul<const IN: usize, const OUT: usize>(
        &self,
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

    fn vector_add<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        use std::arch::aarch64::*;

        let mut result = [0.0; N];
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= N {
            let a_vec = unsafe { vld1q_f32(&a[i]) };
            let b_vec = unsafe { vld1q_f32(&b[i]) };
            let sum_vec = unsafe { vaddq_f32(a_vec, b_vec) };
            unsafe { vst1q_f32(&mut result[i], sum_vec) };
            i += 4;
        }

        // Handle remaining elements
        for j in i..N {
            result[j] = a[j] + b[j];
        }

        result
    }

    fn vector_sub<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        use std::arch::aarch64::*;

        let mut result = [0.0; N];
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= N {
            let a_vec = unsafe { vld1q_f32(&a[i]) };
            let b_vec = unsafe { vld1q_f32(&b[i]) };
            let diff_vec = unsafe { vsubq_f32(a_vec, b_vec) };
            unsafe { vst1q_f32(&mut result[i], diff_vec) };
            i += 4;
        }

        // Handle remaining elements
        for j in i..N {
            result[j] = a[j] - b[j];
        }

        result
    }

    fn vector_mul<const N: usize>(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        use std::arch::aarch64::*;

        let mut result = [0.0; N];
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= N {
            let a_vec = unsafe { vld1q_f32(&a[i]) };
            let b_vec = unsafe { vld1q_f32(&b[i]) };
            let prod_vec = unsafe { vmulq_f32(a_vec, b_vec) };
            unsafe { vst1q_f32(&mut result[i], prod_vec) };
            i += 4;
        }

        // Handle remaining elements
        for j in i..N {
            result[j] = a[j] * b[j];
        }

        result
    }

    fn vector_scale<const N: usize>(&self, a: &[f32; N], scale: f32) -> [f32; N] {
        use std::arch::aarch64::*;

        let mut result = [0.0; N];
        let scale_vec = unsafe { vdupq_n_f32(scale) };
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= N {
            let a_vec = unsafe { vld1q_f32(&a[i]) };
            let scaled_vec = unsafe { vmulq_f32(a_vec, scale_vec) };
            unsafe { vst1q_f32(&mut result[i], scaled_vec) };
            i += 4;
        }

        // Handle remaining elements
        for j in i..N {
            result[j] = a[j] * scale;
        }

        result
    }
}

/// Matrix-vector multiply using the best SIMD backend for this target and CPU features.
pub fn matrix_vector_mul<const IN: usize, const OUT: usize>(
    input: &Obs<IN>,
    weights: &[[f32; IN]; OUT],
    bias: &[f32; OUT],
) -> Action<OUT> {
    #[cfg(all(feature = "simd_avx2", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return SimdBackend::matrix_vector_mul(&Avx2Backend, input, weights, bias);
        }
    }
    #[cfg(all(feature = "simd_neon", target_arch = "aarch64"))]
    {
        return SimdBackend::matrix_vector_mul(&NeonBackend, input, weights, bias);
    }
    SimdBackend::matrix_vector_mul(&ScalarBackend, input, weights, bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_backend() {
        let backend = ScalarBackend;
        let input = Obs::new([1.0, 2.0]);
        let weights = [[1.0, 0.5], [0.5, 1.0]];
        let bias = [0.1, 0.2];

        let output = backend.matrix_vector_mul(&input, &weights, &bias);
        assert_eq!(output.as_slice(), [2.1, 2.7]);
    }

    #[test]
    fn test_vector_operations() {
        let backend = ScalarBackend;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];

        let sum = backend.vector_add(&a, &b);
        assert_eq!(sum, [5.0, 7.0, 9.0]);

        let diff = backend.vector_sub(&a, &b);
        assert_eq!(diff, [-3.0, -3.0, -3.0]);

        let prod = backend.vector_mul(&a, &b);
        assert_eq!(prod, [4.0, 10.0, 18.0]);

        let scaled = backend.vector_scale(&a, 2.0);
        assert_eq!(scaled, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_backend_selection() {
        let out = matrix_vector_mul(&Obs::<1>::new([1.0]), &[[1.0]], &[0.0]);
        assert!((out.as_slice()[0] - 1.0).abs() < 1e-6);
    }
}
