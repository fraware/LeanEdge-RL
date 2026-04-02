import Lake
open Lake DSL

package leanrl_specs where
  version := v!"0.1.0"

@[default_target]
lean_lib PolicySpec where
  roots := #[`PolicySpec]
