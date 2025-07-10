use thiserror::Error;

/// Error types for LeanEdge-RL
#[derive(Error, Debug, Clone, PartialEq)]
pub enum Error {
    #[error("Invalid weights data: {0}")]
    InvalidWeights(String),
    
    #[error("Invalid observation size: expected {expected}, got {actual}")]
    InvalidObsSize { expected: usize, actual: usize },
    
    #[error("Invalid action size: expected {expected}, got {actual}")]
    InvalidActionSize { expected: usize, actual: usize },
    
    #[error("Safety invariant violation: {0}")]
    InvariantViolation(String),
    
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Algorithm not supported: {0}")]
    UnsupportedAlgorithm(String),
    
    #[error("SIMD feature not available: {0}")]
    SimdNotAvailable(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for LeanEdge-RL operations
pub type Result<T> = core::result::Result<T, Error>;

/// FFI error codes (mirror errno.h)
pub mod ffi {
    pub const LR_OK: i32 = 0;
    pub const LR_EBADWEIGHTS: i32 = -1;
    pub const LR_EINVSIZE: i32 = -2;
    pub const LR_EINVARIANT: i32 = -3;
    pub const LR_EOUTOFMEM: i32 = -4;
    pub const LR_EINTERNAL: i32 = -5;
    
    /// Convert Rust error to FFI error code
    pub fn error_to_code(err: &crate::Error) -> i32 {
        match err {
            crate::Error::InvalidWeights(_) => LR_EBADWEIGHTS,
            crate::Error::InvalidObsSize { .. } | crate::Error::InvalidActionSize { .. } => LR_EINVSIZE,
            crate::Error::InvariantViolation(_) => LR_EINVARIANT,
            crate::Error::OutOfMemory(_) => LR_EOUTOFMEM,
            _ => LR_EINTERNAL,
        }
    }
    
    /// Convert FFI error code to Rust error
    pub fn code_to_error(code: i32) -> crate::Error {
        match code {
            LR_EBADWEIGHTS => crate::Error::InvalidWeights("FFI: Bad weights".to_string()),
            LR_EINVSIZE => crate::Error::InvalidObsSize { expected: 0, actual: 0 },
            LR_EINVARIANT => crate::Error::InvariantViolation("FFI: Invariant violation".to_string()),
            LR_EOUTOFMEM => crate::Error::OutOfMemory("FFI: Out of memory".to_string()),
            _ => crate::Error::Internal(format!("FFI: Unknown error code {}", code)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_conversion() {
        let err = Error::InvalidWeights("test".to_string());
        let code = ffi::error_to_code(&err);
        assert_eq!(code, ffi::LR_EBADWEIGHTS);
        
        let converted_err = ffi::code_to_error(code);
        assert!(matches!(converted_err, Error::InvalidWeights(_)));
    }
} 