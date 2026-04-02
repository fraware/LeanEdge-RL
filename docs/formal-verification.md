# Formal verification (Lean 4)

LeanEdge-RL is designed so that **safety properties** can be stated and proved in **Lean 4** alongside the Rust runtime. The connection is incremental: in-repo specs exist today; full proof coverage and policy export are **roadmap** work.

## What is in this repository

| Path | Purpose |
|------|---------|
| [lean/lakefile.lean](../lean/lakefile.lean) | Lake workspace definition |
| [lean/PolicySpec.lean](../lean/PolicySpec.lean) | Starter module for theorems (expand over time) |
| [lean/lean-toolchain](../lean/lean-toolchain) | Pinned Lean version for reproducible `lake build` |

CI runs `lake build` in `lean/` on pushes and pull requests (see [.github/workflows/ci.yml](../.github/workflows/ci.yml)).

## Policy export and Rust workspace

The Rust workspace **no longer** includes `lake-packages/leanrl_export` by default (that path was missing and broke `cargo metadata`). Choose one of:

### Option A — Git submodule (split repo)

1. Add a submodule at `lake-packages/leanrl_export` containing a Lake (and optionally Rust) package that emits policy artifacts, hashes, or FFI stubs.
2. If the export tool is a **Rust** crate, add `"lake-packages/leanrl_export"` back to `[workspace].members` in the root [Cargo.toml](../Cargo.toml).

### Option B — In-repo Lake only

Keep all Lean sources under [lean/](../lean/). Use `lake build` locally and in CI. Pass proof or artifact hashes into tooling manually, for example:

```bash
cargo run -p leanrl-bundle -- generate --proof-hash <sha256-or-git-rev>
```

The bundle CLI records `proof_hash` in metadata when provided.

## Local development

Install [Elan](https://github.com/leanprover/elan) and run:

```bash
cd lean
lake build
```

Elan reads `lean-toolchain` and fetches the matching Lean 4 toolchain.

## Relation to Rust

- **Rust** (`leanrl_core`) is the executable runtime and C API.  
- **Lean** should eventually **specify** or **refine** invariants that the Rust side enforces (e.g. action bounds, weight validity).  
- Until proofs are complete, use `sorry` or small tautologies only in development branches; prefer merging meaningful lemmas or commented proof obligations.

## Further reading

- [README.md](../README.md) — project overview and roadmap  
- [CONTRIBUTING.md](../CONTRIBUTING.md) — contributor workflow and Lean expectations  
