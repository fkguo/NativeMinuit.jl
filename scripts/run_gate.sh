#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-2.1-or-later
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/benchmark"
# Prepare the local source and resolved environment before any timed operation.
JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no --project=. -e \
    'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
exec julia --startup-file=no --project=. "$REPO_ROOT/scripts/run_perf.jl" \
    --config "$REPO_ROOT/benchmark/perf-config.toml" "$@"
