//! Core RL runtime. `unsafe` is restricted to `ffi.rs` (C API) and SIMD/intrinsic sites in
//! `simd.rs` (and arch-specific code); see `CONTRIBUTING.md` and `deny.toml` / CI allowlists.
#![deny(unsafe_op_in_unsafe_fn)]
// Stable C API uses `extern "C" fn` (not `unsafe fn`); clippy still sees raw pointer use inside.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::needless_range_loop)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), no_implicit_prelude)]

// Safety gate: Ensure environment state fits in 1MB
const _ENV_STATE_SIZE_CHECK: () = {
    const ENV_STATE_SIZE: usize = core::mem::size_of::<crate::env::EnvState>();
    assert!(ENV_STATE_SIZE < 1_048_576, "EnvState must be < 1MB");
};

pub mod action;
pub mod algorithms;
pub mod env;
pub mod error;
pub mod ffi;
pub mod obs;
pub mod simd;

// Re-export main types
pub use action::Action;
pub use env::Env;
pub use error::{Error, Result};
pub use obs::Obs;

// Type aliases for common dimensions
pub type Dim = usize;

/// Environment trait for different RL algorithms
pub trait Environment<const OBS_DIM: usize, const ACTION_DIM: usize> {
    /// Reset the environment with initial observation
    fn reset(&mut self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM>;

    /// Step the environment with new observation
    fn step(&mut self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM>;

    /// Get current environment state
    fn state(&self) -> &env::EnvState;

    /// Set environment state (for testing/debugging)
    fn set_state(&mut self, state: env::EnvState);
}

/// Policy trait for different RL algorithms
pub trait Policy<const OBS_DIM: usize, const ACTION_DIM: usize> {
    /// Compute action from observation
    fn act(&self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM>;

    /// Update policy weights
    fn update_weights(&mut self, weights: &[u8]) -> Result<()>;

    /// Get policy weights for serialization
    fn get_weights(&self) -> Result<Vec<u8>>;

    /// Stable name for logging, SBOM, and bundle metadata
    fn algorithm_name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(path_statements)]
    fn test_env_state_size() {
        // This test ensures the safety gate is working
        _ENV_STATE_SIZE_CHECK;
    }

    #[test]
    fn test_basic_types() {
        let obs = Obs::new([1.0, 2.0, 3.0, 4.0]);
        let action = Action::new([0.5, -0.5]);

        assert_eq!(obs.as_slice(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(action.as_slice(), [0.5, -0.5]);
    }
}
