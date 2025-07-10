// This is the only file allowed to use unsafe code in the entire crate
// All unsafe code must be audited and documented

use crate::{
    error::{Error, Result, ffi as error_ffi},
    env::{Env, EnvState},
    obs::Obs,
    action::Action,
};

use std::ffi::c_void;
use std::ptr;

/// Opaque environment handle for C API
pub struct lr_env {
    env: Option<Env<'static, 4, 2>>, // Using fixed dimensions for C API
    weights: Vec<u8>, // Keep weights alive
}

/// C API: Initialize environment with weights
#[no_mangle]
pub extern "C" fn lr_init(
    weights: *const u8,
    len: usize,
    out: *mut *mut lr_env,
) -> i32 {
    // Safety: Check for null pointers
    if weights.is_null() || out.is_null() {
        return error_ffi::LR_EBADWEIGHTS;
    }
    
    // Safety: Validate input slice
    let weights_slice = unsafe { std::slice::from_raw_parts(weights, len) };
    
    // Create weights vector to keep data alive
    let weights_vec = weights_slice.to_vec();
    
    // Create environment
    let env_result = Env::<4, 2>::from_weights(&weights_vec);
    match env_result {
        Ok(env) => {
            // Allocate environment handle
            let env_handle = Box::new(lr_env {
                env: Some(env),
                weights: weights_vec,
            });
            
            // Safety: Write pointer to output
            unsafe {
                *out = Box::into_raw(env_handle);
            }
            
            error_ffi::LR_OK
        }
        Err(_) => error_ffi::LR_EBADWEIGHTS,
    }
}

/// C API: Reset environment with initial observation
#[no_mangle]
pub extern "C" fn lr_reset(
    env: *mut lr_env,
    obs: *const f32,
    action: *mut f32,
) -> i32 {
    // Safety: Check for null pointers
    if env.is_null() || obs.is_null() || action.is_null() {
        return error_ffi::LR_EINVSIZE;
    }
    
    // Safety: Dereference environment handle
    let env_handle = unsafe { &mut *env };
    let env_ref = match &mut env_handle.env {
        Some(env) => env,
        None => return error_ffi::LR_EINTERNAL,
    };
    
    // Safety: Create observation from C array
    let obs_slice = unsafe { std::slice::from_raw_parts(obs, 4) };
    let obs_result = Obs::<4>::from_slice(obs_slice);
    let obs = match obs_result {
        Ok(obs) => obs,
        Err(_) => return error_ffi::LR_EINVSIZE,
    };
    
    // Reset environment
    let action_result = env_ref.reset(&obs);
    
    // Safety: Write action to output array
    let action_slice = unsafe { std::slice::from_raw_parts_mut(action, 2) };
    action_slice.copy_from_slice(action_result.as_slice());
    
    error_ffi::LR_OK
}

/// C API: Step environment with new observation
#[no_mangle]
pub extern "C" fn lr_step(
    env: *mut lr_env,
    obs: *const f32,
    action: *mut f32,
) -> i32 {
    // Safety: Check for null pointers
    if env.is_null() || obs.is_null() || action.is_null() {
        return error_ffi::LR_EINVSIZE;
    }
    
    // Safety: Dereference environment handle
    let env_handle = unsafe { &mut *env };
    let env_ref = match &mut env_handle.env {
        Some(env) => env,
        None => return error_ffi::LR_EINTERNAL,
    };
    
    // Safety: Create observation from C array
    let obs_slice = unsafe { std::slice::from_raw_parts(obs, 4) };
    let obs_result = Obs::<4>::from_slice(obs_slice);
    let obs = match obs_result {
        Ok(obs) => obs,
        Err(_) => return error_ffi::LR_EINVSIZE,
    };
    
    // Step environment
    let action_result = env_ref.step(&obs);
    
    // Safety: Write action to output array
    let action_slice = unsafe { std::slice::from_raw_parts_mut(action, 2) };
    action_slice.copy_from_slice(action_result.as_slice());
    
    error_ffi::LR_OK
}

/// C API: Free environment handle
#[no_mangle]
pub extern "C" fn lr_free(env: *mut lr_env) {
    if !env.is_null() {
        // Safety: Drop the environment handle
        unsafe {
            let _ = Box::from_raw(env);
        }
    }
}

/// C API: Get environment state (for debugging/testing)
#[no_mangle]
pub extern "C" fn lr_get_state(
    env: *const lr_env,
    step_count: *mut u64,
    episode_count: *mut u64,
) -> i32 {
    // Safety: Check for null pointers
    if env.is_null() || step_count.is_null() || episode_count.is_null() {
        return error_ffi::LR_EINTERNAL;
    }
    
    // Safety: Dereference environment handle
    let env_handle = unsafe { &*env };
    let env_ref = match &env_handle.env {
        Some(env) => env,
        None => return error_ffi::LR_EINTERNAL,
    };
    
    let state = env_ref.state();
    
    // Safety: Write state to output pointers
    unsafe {
        *step_count = state.step_count;
        *episode_count = state.episode_count;
    }
    
    error_ffi::LR_OK
}

/// C API: Check safety invariant
#[no_mangle]
pub extern "C" fn lr_check_invariant(
    env: *const lr_env,
    obs: *const f32,
    action: *const f32,
) -> i32 {
    // Safety: Check for null pointers
    if env.is_null() || obs.is_null() || action.is_null() {
        return error_ffi::LR_EINVARIANT;
    }
    
    // Safety: Dereference environment handle
    let env_handle = unsafe { &*env };
    let env_ref = match &env_handle.env {
        Some(env) => env,
        None => return error_ffi::LR_EINTERNAL,
    };
    
    // Safety: Create observation and action from C arrays
    let obs_slice = unsafe { std::slice::from_raw_parts(obs, 4) };
    let obs_result = Obs::<4>::from_slice(obs_slice);
    let obs = match obs_result {
        Ok(obs) => obs,
        Err(_) => return error_ffi::LR_EINVSIZE,
    };
    
    let action_slice = unsafe { std::slice::from_raw_parts(action, 2) };
    let action_result = Action::<2>::from_slice(action_slice);
    let action = match action_result {
        Ok(action) => action,
        Err(_) => return error_ffi::LR_EINVSIZE,
    };
    
    // Check invariant
    match env_ref.check_invariant(&obs, &action) {
        Ok(_) => error_ffi::LR_OK,
        Err(_) => error_ffi::LR_EINVARIANT,
    }
}

/// C API: Update environment weights
#[no_mangle]
pub extern "C" fn lr_update_weights(
    env: *mut lr_env,
    weights: *const u8,
    len: usize,
) -> i32 {
    // Safety: Check for null pointers
    if env.is_null() || weights.is_null() {
        return error_ffi::LR_EBADWEIGHTS;
    }
    
    // Safety: Dereference environment handle
    let env_handle = unsafe { &mut *env };
    let env_ref = match &mut env_handle.env {
        Some(env) => env,
        None => return error_ffi::LR_EINTERNAL,
    };
    
    // Safety: Create weights slice
    let weights_slice = unsafe { std::slice::from_raw_parts(weights, len) };
    
    // Update weights
    match env_ref.update_weights(weights_slice) {
        Ok(_) => {
            // Update stored weights
            env_handle.weights = weights_slice.to_vec();
            error_ffi::LR_OK
        }
        Err(_) => error_ffi::LR_EBADWEIGHTS,
    }
}

/// C API: Get environment weights
#[no_mangle]
pub extern "C" fn lr_get_weights(
    env: *const lr_env,
    weights: *mut u8,
    max_len: usize,
    actual_len: *mut usize,
) -> i32 {
    // Safety: Check for null pointers
    if env.is_null() || weights.is_null() || actual_len.is_null() {
        return error_ffi::LR_EBADWEIGHTS;
    }
    
    // Safety: Dereference environment handle
    let env_handle = unsafe { &*env };
    let env_ref = match &env_handle.env {
        Some(env) => env,
        None => return error_ffi::LR_EINTERNAL,
    };
    
    // Get weights
    let weights_result = env_ref.get_weights();
    let weights_vec = match weights_result {
        Ok(weights) => weights,
        Err(_) => return error_ffi::LR_EBADWEIGHTS,
    };
    
    // Safety: Write weights to output buffer
    let actual_size = weights_vec.len().min(max_len);
    unsafe {
        std::ptr::copy_nonoverlapping(weights_vec.as_ptr(), weights, actual_size);
        *actual_len = weights_vec.len();
    }
    
    error_ffi::LR_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ffi_init_and_free() {
        let mut weights = vec![0u8]; // TabularQLearning
        weights.extend(vec![1.0f32.to_le_bytes().to_vec()].concat());
        
        let mut env_ptr: *mut lr_env = ptr::null_mut();
        let result = lr_init(weights.as_ptr(), weights.len(), &mut env_ptr);
        
        assert_eq!(result, error_ffi::LR_OK);
        assert!(!env_ptr.is_null());
        
        // Free environment
        lr_free(env_ptr);
    }
    
    #[test]
    fn test_ffi_reset_and_step() {
        let mut weights = vec![0u8]; // TabularQLearning
        weights.extend(vec![1.0f32.to_le_bytes().to_vec()].concat());
        
        let mut env_ptr: *mut lr_env = ptr::null_mut();
        lr_init(weights.as_ptr(), weights.len(), &mut env_ptr);
        
        let obs = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let mut action = [0.0f32; 2];
        
        let result = lr_reset(env_ptr, obs.as_ptr(), action.as_mut_ptr());
        assert_eq!(result, error_ffi::LR_OK);
        
        let result = lr_step(env_ptr, obs.as_ptr(), action.as_mut_ptr());
        assert_eq!(result, error_ffi::LR_OK);
        
        lr_free(env_ptr);
    }
    
    #[test]
    fn test_ffi_null_pointer_handling() {
        let result = lr_init(ptr::null(), 0, ptr::null_mut());
        assert_eq!(result, error_ffi::LR_EBADWEIGHTS);
        
        lr_free(ptr::null_mut()); // Should not crash
    }
} 