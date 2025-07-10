use cxx::UniquePtr;
use leanrl_core::{Env, Obs, Action, error::Result};

#[cxx::bridge(namespace = "leanrl")]
mod ffi {
    // Rust types that can be shared with C++
    unsafe extern "C++" {
        include!("leanrl_cshim/include/leanrl.hpp");
        
        type Env4x2;
        type Obs4;
        type Action2;
        
        fn new_env4x2(weights: &[u8]) -> UniquePtr<Env4x2>;
        fn reset(self: &Env4x2, obs: &Obs4) -> UniquePtr<Action2>;
        fn step(self: &Env4x2, obs: &Obs4) -> UniquePtr<Action2>;
        fn get_state(self: &Env4x2) -> (u64, u64); // (step_count, episode_count)
        fn check_invariant(self: &Env4x2, obs: &Obs4, action: &Action2) -> bool;
        fn update_weights(self: &mut Env4x2, weights: &[u8]) -> bool;
        fn get_weights(self: &Env4x2) -> Vec<u8>;
        
        fn new_obs4(data: [f32; 4]) -> UniquePtr<Obs4>;
        fn get_data(self: &Obs4) -> [f32; 4];
        fn set_data(self: &mut Obs4, data: [f32; 4]);
        
        fn new_action2(data: [f32; 2]) -> UniquePtr<Action2>;
        fn get_data(self: &Action2) -> [f32; 2];
        fn set_data(self: &mut Action2, data: [f32; 2]);
    }
    
    // C++ types that can be shared with Rust
    extern "Rust" {
        type RustEnv4x2;
        type RustObs4;
        type RustAction2;
        
        fn create_env4x2(weights: &[u8]) -> Result<Box<RustEnv4x2>>;
        fn reset_env(env: &mut RustEnv4x2, obs: &RustObs4) -> RustAction2;
        fn step_env(env: &mut RustEnv4x2, obs: &RustObs4) -> RustAction2;
        fn get_env_state(env: &RustEnv4x2) -> (u64, u64);
        fn check_env_invariant(env: &RustEnv4x2, obs: &RustObs4, action: &RustAction2) -> Result<()>;
        fn update_env_weights(env: &mut RustEnv4x2, weights: &[u8]) -> Result<()>;
        fn get_env_weights(env: &RustEnv4x2) -> Result<Vec<u8>>;
        
        fn create_obs4(data: [f32; 4]) -> RustObs4;
        fn get_obs_data(obs: &RustObs4) -> [f32; 4];
        fn set_obs_data(obs: &mut RustObs4, data: [f32; 4]) -> Result<()>;
        
        fn create_action2(data: [f32; 2]) -> RustAction2;
        fn get_action_data(action: &RustAction2) -> [f32; 2];
        fn set_action_data(action: &mut RustAction2, data: [f32; 2]) -> Result<()>;
    }
}

// Rust wrapper types
pub struct RustEnv4x2 {
    env: Env<'static, 4, 2>,
    weights: Vec<u8>,
}

pub struct RustObs4(Obs<4>);
pub struct RustAction2(Action<2>);

// Implementation of Rust types
impl RustEnv4x2 {
    fn new(weights: &[u8]) -> Result<Self> {
        let env = Env::<4, 2>::from_weights(weights)?;
        Ok(Self {
            env,
            weights: weights.to_vec(),
        })
    }
}

impl RustObs4 {
    fn new(data: [f32; 4]) -> Self {
        Self(Obs::new(data))
    }
}

impl RustAction2 {
    fn new(data: [f32; 2]) -> Self {
        Self(Action::new(data))
    }
}

// CXX bridge implementations
pub fn create_env4x2(weights: &[u8]) -> Result<Box<RustEnv4x2>> {
    Ok(Box::new(RustEnv4x2::new(weights)?))
}

pub fn reset_env(env: &mut RustEnv4x2, obs: &RustObs4) -> RustAction2 {
    RustAction2(env.env.reset(&obs.0))
}

pub fn step_env(env: &mut RustEnv4x2, obs: &RustObs4) -> RustAction2 {
    RustAction2(env.env.step(&obs.0))
}

pub fn get_env_state(env: &RustEnv4x2) -> (u64, u64) {
    let state = env.env.state();
    (state.step_count, state.episode_count)
}

pub fn check_env_invariant(env: &RustEnv4x2, obs: &RustObs4, action: &RustAction2) -> Result<()> {
    env.env.check_invariant(&obs.0, &action.0)
}

pub fn update_env_weights(env: &mut RustEnv4x2, weights: &[u8]) -> Result<()> {
    env.env.update_weights(weights)?;
    env.weights = weights.to_vec();
    Ok(())
}

pub fn get_env_weights(env: &RustEnv4x2) -> Result<Vec<u8>> {
    env.env.get_weights()
}

pub fn create_obs4(data: [f32; 4]) -> RustObs4 {
    RustObs4::new(data)
}

pub fn get_obs_data(obs: &RustObs4) -> [f32; 4] {
    *obs.0.as_array()
}

pub fn set_obs_data(obs: &mut RustObs4, data: [f32; 4]) -> Result<()> {
    obs.0 = Obs::new(data);
    Ok(())
}

pub fn create_action2(data: [f32; 2]) -> RustAction2 {
    RustAction2::new(data)
}

pub fn get_action_data(action: &RustAction2) -> [f32; 2] {
    *action.0.as_array()
}

pub fn set_action_data(action: &mut RustAction2, data: [f32; 2]) -> Result<()> {
    action.0 = Action::new(data);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rust_wrapper_creation() {
        let mut weights = vec![0u8]; // TabularQLearning
        weights.extend(vec![1.0f32.to_le_bytes().to_vec()].concat());
        
        let env = RustEnv4x2::new(&weights);
        assert!(env.is_ok());
        
        let obs = RustObs4::new([1.0, 2.0, 3.0, 4.0]);
        let action = RustAction2::new([0.5, -0.3]);
        
        assert_eq!(get_obs_data(&obs), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(get_action_data(&action), [0.5, -0.3]);
    }
} 