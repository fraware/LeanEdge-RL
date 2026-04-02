//! C/C++ integration helpers: re-exports the core runtime. Headers live under `include/`.
//! Prefer linking `leanrl_core` as `cdylib`/`staticlib` and calling the `lr_*` C API in `ffi.rs`,
//! or wrapping those symbols from C++.

pub use leanrl_core::*;
