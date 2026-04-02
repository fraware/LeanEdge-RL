#!/usr/bin/env bash
# Fail if `unsafe` appears outside audited core sources (see CONTRIBUTING.md).
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"
ALLOWED='^core/src/(ffi|simd)\.rs$'
while IFS= read -r -d '' file; do
  rel="${file#./}"
  if [[ "$rel" =~ $ALLOWED ]]; then
    continue
  fi
  echo "Disallowed unsafe in: $rel (only core/src/ffi.rs and core/src/simd.rs are permitted)"
  exit 1
done < <(grep -rlZ --include='*.rs' -E '\bunsafe\b' core/src || true)
echo "Unsafe allowlist check passed."
