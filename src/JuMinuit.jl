# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 fkguo and JuMinuit.jl contributors
#
# Derivative work of C++ Minuit2 (GooFit/Minuit2 @ v6.24.0); see LICENSE
# and docs/UPSTREAM.md.

"""
    JuMinuit

Native-Julia port of the C++ Minuit2 function-minimization library, the
algorithm at the heart of every HEP fit. Targets drop-in replacement of
the iminuit/IMinuit.jl stack with C++-comparable performance.

Status: **Phase 0 (proof of concept)** — see [`ROADMAP.md`](../ROADMAP.md).
Phase 0 ships unconstrained MIGRAD with numerical gradient and
`Strategy(0)` only. Bounds, fixed parameters, MINOS, contours, and HESSE
land in Phase 1.

The implementation mirrors `reference/Minuit2_cpp/` (pinned to
`57dc936`, v6.24.0). Each src/ file maps 1-to-1 to a C++ translation
unit so audits diff cleanly. See `docs/PORTING.md` for the mapping.
"""
module JuMinuit

using LinearAlgebra
using Logging
using Printf

include("precision.jl")
include("strategy.jl")

# Phase 0 public surface (will grow as files are added).
export MachinePrecision
export Strategy

end # module JuMinuit
