#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Phase 0 §3.4 evidence-gate driver.
#
# Activates scripts/ environment (which dev-paths the local NativeMinuit
# package), instantiates deps if missing, runs scripts/run_perf.jl
# against benchmark/perf-config.toml, prints the verdict.
#
# Usage:  scripts/run_gate.sh [--save-baseline]

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
CONFIG="$REPO_ROOT/benchmark/perf-config.toml"

cd "$SCRIPTS_DIR"

echo ">>> Instantiating scripts/ environment (BenchmarkTools, JSON3, NativeMinuit)"
julia --project=. -e 'using Pkg; Pkg.instantiate()'

echo ">>> Running julia-perf gate against $CONFIG"
julia --project=. run_perf.jl --config "$CONFIG" "$@"
