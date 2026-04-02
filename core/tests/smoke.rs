//! Integration smoke tests for workspace loading and `Env` construction.

use leanrl_core::env::Env;
use leanrl_core::obs::Obs;

fn minimal_tabular_weights() -> Vec<u8> {
    let mut w = Vec::new();
    w.push(0u8); // AlgorithmType::TabularQLearning
    w.extend_from_slice(&1u32.to_le_bytes()); // num_states
    w.extend_from_slice(&1u32.to_le_bytes()); // num_actions
    w.extend_from_slice(&0.1f32.to_le_bytes()); // alpha
    w.extend_from_slice(&0.9f32.to_le_bytes()); // gamma
    w
}

#[test]
fn env_from_tabular_weights_roundtrip() {
    let weights = minimal_tabular_weights();
    let mut env = Env::<4, 2>::from_weights(&weights).expect("valid tabular header");
    let obs = Obs::new([0.0f32, 0.0, 0.0, 0.0]);
    let _a = env.reset(&obs);
    let _b = env.step(&obs);
}
