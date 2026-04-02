# Contributing to LeanEdge-RL

Thank you for your interest in contributing. This document describes setup, standards, and the pull-request workflow.

## Development setup

### Prerequisites

- **Rust**: stable with `rustfmt` and `clippy` (pinned in [rust-toolchain.toml](rust-toolchain.toml); `rustup` will pick it up automatically)
- **Git**
- **Optional**: Clang/LLVM for some native dependencies on certain platforms; cross-compilation toolchains for non-host triples
- **Optional**: Lean 4 (`elan`) if you work on [lean/](lean/)
- **Optional**: CMake 3.20+ for [cshim/cpp_tests](cshim/cpp_tests)

### Clone and build

```bash
git clone https://github.com/leanrl/leanedge-rl.git
cd leanedge-rl

cargo build
cargo test --workspace
```

### Feature and target examples

```bash
# SIMD (only on matching architecture)
cargo build -p leanrl_core --features simd_avx2
cargo build -p leanrl_core --features simd_neon   # aarch64 host

# Embedded-style core build
cargo build -p leanrl_core --target thumbv7em-none-eabi --no-default-features
```

## Code standards

### Rust

- Run **`rustfmt`**: `cargo fmt --all`
- Run **`clippy`** as in CI:

  ```bash
  cargo clippy --workspace --all-targets --all-features -- -D warnings
  ```

- Add tests for new behavior (`cargo test -p leanrl_core` or `--workspace`)

### `unsafe` policy

**Any new `unsafe` must:**

1. Be reviewed by two people when policy-critical  
2. Be documented with safety invariants  
3. Prefer MIRI where it applies (`cargo +nightly miri test`)  
4. Land with a clear commit message  

**Allowed files** (enforced in CI via [scripts/verify-unsafe-allowlist.sh](scripts/verify-unsafe-allowlist.sh)):

- `core/src/ffi.rs` — C API  
- `core/src/simd.rs` — SIMD intrinsics  

To add another file, update the script and this list.

On **Windows**, run the allowlist script with Git Bash or WSL: `bash scripts/verify-unsafe-allowlist.sh`.

### C++

- C++17 or as agreed by maintainers  
- The GTest harness under [cshim/cpp_tests](cshim/cpp_tests) is a starting point; production integration should link `leanrl_core` and/or the `lr_*` API from your build system  

## Testing

### Unit and integration tests

```bash
cargo test --workspace
cargo test -p leanrl_core
cargo test -p leanrl_core --test smoke
```

### Crate-specific features

```bash
cargo test -p leanrl_core --features simd_avx2
```

### Nextest (optional)

```bash
cargo install cargo-nextest --locked
cargo nextest run --workspace
```

### Benchmarks

```bash
cargo bench -p leanrl_core
```

There is no `performance` bench target; Criterion benches live under [core/benches](core/benches).

### Security

```bash
cargo audit
# Optional: cargo install cargo-deny --locked && cargo deny check
cargo install cargo-geiger --locked   # optional unsafe survey
cargo +nightly miri test -p leanrl_core   # if applicable
```

## Formal verification (Lean 4)

- Specifications live under [lean/](lean/lakefile.lean) (`lake build`).  
- Optional export tooling can live in a submodule at `lake-packages/leanrl_export`; see [docs/formal-verification.md](docs/formal-verification.md).  
- Keep CI proof time reasonable (on the order of minutes for the current small spec).  

Example skeleton (names are illustrative):

```lean
theorem action_bounds_invariant (s : EnvState) (obs : Obs) (action : Action) :
  inv s → inv (step s obs action) := by
  sorry
```

## Pull requests

### Before opening a PR

1. Branch from `main`  
2. Add or update tests and docs  
3. Run locally:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo audit
bash scripts/verify-unsafe-allowlist.sh
```

Cross-builds (optional locally, covered in CI):

```bash
cargo build --target x86_64-unknown-linux-gnu
cargo build --target aarch64-unknown-linux-gnu
cargo build -p leanrl_core --target thumbv7em-none-eabi --no-default-features
```

### PR checklist

- Clear description and linked issues where relevant  
- CI green  
- Maintainer review  
- Note performance or ABI impact when relevant  

### Commits

Prefer [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat(core): ...
fix(ffi): ...
docs: ...
test: ...
```

## API and ABI

### Rust API

Breaking changes should follow semver and, when needed, a short migration note in the PR description or the GitHub release.

### C API (`lr_*`)

Treat the C ABI as stable across minor versions unless explicitly bumped. Add tests for new entry points.

## Performance and memory (targets)

Documented targets (see README) are **goals**. If you change hot paths or allocations, mention measured impact in the PR.

## Security

- No secrets in the repo  
- Validate untrusted inputs at FFI boundaries  
- Prefer `cargo deny` / `cargo audit` clean in CI  

Report sensitive issues through the project’s security contact if published; otherwise use GitHub private reporting if enabled.

## Releases

### Checklist

1. Tests and clippy clean on supported targets  
2. Release notes summarized in the GitHub release (or PR) as appropriate  
3. Version bumped in workspace `Cargo.toml` as appropriate  
4. Tag `v*.*.*` triggers [.github/workflows/release.yml](.github/workflows/release.yml) (binary tarball for `leanrl-bundle` on Linux)  

### Bundle commands

```bash
cargo run -p leanrl-bundle -- generate
cargo run -p leanrl-bundle -- generate --sign    # when signing is configured
cargo run -p leanrl-bundle -- verify path/to/leanrl_bundle_*.zip
```

## Getting help

- **Issues**: bugs and features  
- **Discussions**: design questions (if enabled)  
- **RFCs**: `/rfcs` for large changes, if present  

## Code of conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).

## License

Contributions are licensed under **MIT OR Apache-2.0**, in line with [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
