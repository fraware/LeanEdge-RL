# Contributing to LeanEdge-RL

Thank you for your interest in contributing to LeanEdge-RL! This document outlines the development guidelines and contribution process.

## Development Setup

### Prerequisites

- Rust 1.78+ (stable)
- Clang 18+ (for C/C++ components)
- Lean 4 (for formal verification)
- Git

### Building from Source

```bash
# Clone the repository
git clone https://github.com/leanrl/leanedge-rl.git
cd leanedge-rl

# Build all crates
cargo build

# Run tests
cargo test

# Build with SIMD support
cargo build --features simd_avx2

# Build for embedded targets
cargo build --target thumbv7em-none-eabi --no-default-features
```

## Code Standards

### Rust Code

- Follow Rust style guidelines (enforced by `rustfmt`)
- Use `#![forbid(unsafe_code)]` in all modules except `ffi.rs`
- Write comprehensive tests for all new functionality
- Use `cargo clippy` for additional linting

### Unsafe Code Policy

**CRITICAL**: Any new unsafe code requires:

1. Two independent reviewers
2. MIRI trace proof
3. Documentation explaining the safety invariants
4. Audit trail in the commit message

Only `core/src/ffi.rs` is allowed to contain unsafe code, and it must be thoroughly audited.

### C++ Code

- Follow modern C++17 standards
- Use RAII and smart pointers
- Provide comprehensive error handling
- Include unit tests using Google Test

## Testing

### Unit Tests

```bash
# Run all tests
cargo test

# Run tests with nextest (faster)
cargo install nextest
cargo nextest run

# Run tests for specific crate
cargo test -p leanrl_core
```

### Integration Tests

```bash
# Run integration tests
cargo test --test integration

# Run with specific features
cargo test --features simd_avx2
```

### Performance Tests

```bash
# Run benchmarks
cargo bench

# Performance regression testing
cargo bench --bench performance
```

### Security Tests

```bash
# Security audit
cargo audit

# Check for unsafe code
cargo install cargo-geiger
cargo geiger

# MIRI testing
cargo +nightly miri test
```

## Formal Verification

### Lean 4 Integration

- All safety invariants must be proven in Lean 4
- Proofs must check in ≤ 90 seconds in CI
- Use the `lake-packages/leanrl_export` package for policy export

### Writing Proofs

```lean
-- Example safety invariant proof
theorem action_bounds_invariant (s : EnvState) (obs : Obs) (action : Action) :
  inv s → inv (step s obs action) := by
  -- Proof implementation
  sorry
```

## Pull Request Process

### Before Submitting

1. **Fork and clone** the repository
2. **Create a feature branch** from `main`
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Run all checks** locally

### Required Checks

```bash
# Format code
cargo fmt

# Lint code
cargo clippy

# Run tests
cargo test

# Security audit
cargo audit

# Check unsafe code
grep -r "unsafe" core/src/ --exclude=ffi.rs

# Build all targets
cargo build --target x86_64-unknown-linux-gnu
cargo build --target aarch64-unknown-linux-gnu
cargo build --target thumbv7em-none-eabi --no-default-features
```

### PR Requirements

1. **Clear description** of changes
2. **Link to related issues** or RFCs
3. **All CI checks passing**
4. **Code review approval** from maintainers
5. **Performance impact assessment** (if applicable)

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(core): add new RL algorithm implementation
fix(ffi): resolve memory leak in C API
docs: update API documentation
test: add integration tests for SIMD backends
```

## API Changes

### Public API Modifications

Any changes to public APIs require:

1. **RFC process** for breaking changes
2. **Semantic versioning** compliance
3. **Migration guide** for users
4. **Backward compatibility** considerations

### C API Stability

The C API (`lr_*` functions) must maintain ABI stability:

- No breaking changes without major version bump
- All changes must be backward compatible
- Comprehensive testing across platforms

## Performance Requirements

### Latency Targets

- P99 ≤ 100 µs on Cortex-A53 @ 1.4 GHz
- P95 ≤ 120 µs in CI performance gates
- Sub-millisecond performance on all supported platforms

### Memory Footprint

- Library code ≤ 350 kB
- Static weights ≤ 256 kB
- RAM usage ≤ 1 MB

## Security Guidelines

### Supply Chain Security

- All dependencies must be auditable
- SBOM generation for every release
- Sigstore signing for all artifacts
- TPM attestation for critical deployments

### Code Security

- No hardcoded secrets
- Secure random number generation
- Input validation and sanitization
- Memory safety guarantees

## Release Process

### Pre-release Checklist

1. **All tests passing** on all targets
2. **Performance benchmarks** within targets
3. **Security audit** completed
4. **Documentation** updated
5. **Compliance bundle** generated
6. **Formal proofs** verified

### Release Steps

```bash
# Generate compliance bundle
cargo run --bin leanrl-bundle -- generate --sign --tpm-attest

# Verify bundle integrity
cargo run --bin leanrl-bundle -- verify leanrl_bundle_*.zip

# Tag release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

## Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **RFCs**: Submit RFCs for major changes in `/rfcs`
- **Security**: Report security issues privately to security@leanrl.org

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and inclusive in all interactions.

## License

By contributing to LeanEdge-RL, you agree that your contributions will be licensed under the MIT OR Apache-2.0 license.
