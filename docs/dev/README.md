# Development-process archive (point-in-time snapshot)

> **Frozen as of 2026-06-02 (≈ v0.3.1).** The files in this directory are
> development-process records — fidelity audits, design notes, the original
> phased roadmap — kept for provenance and as the engineering audit trail.
> They are **not** maintained as current documentation: nothing here knows
> about later releases (v0.4.0 `find_deeper_minimum` rework, v0.5.0
> `find_solution_modes` overhaul + the iminuit-parity contour family, v0.5.1
> `extremize`/`profile_band` + `mcmc_sample`, …). For the current state of
> the package, see the [CHANGELOG](../../CHANGELOG.md) and the
> [manual](https://fkguo.github.io/NativeMinuit.jl/dev).

| File | What it is |
| --- | --- |
| [`ROADMAP.md`](ROADMAP.md) | The original phased development plan (Phases 0–3), with its own status stamps; all phases complete as of v0.3.0. |
| [`DESIGN.md`](DESIGN.md) | Architectural decision log for the v0.3-era core — the *why* behind each choice, recorded as it was made. |
| [`DEFERRED.md`](DEFERRED.md) | Features knowingly postponed, each with the reason and what would change it. Largely still accurate, but check the CHANGELOG before trusting any single entry. |
| [`CPP_FIDELITY_AUDIT.md`](CPP_FIDELITY_AUDIT.md) | Line-by-line, branch-by-branch fidelity audit of every ported algorithm against upstream C++ Minuit2 v6.24.0 (2026-05-30; line cites pinned to a fixed commit). |
| [`GAP_AUDIT.md`](GAP_AUDIT.md) | Feature-gap inventory vs C++ Minuit2 v6.24.0 (2026-05-27); supplements ROADMAP §9. |
| [`DAVIDON_CXX_AUDIT.md`](DAVIDON_CXX_AUDIT.md) | Investigation record: DFP iteration-1 EDM divergence vs iminuit on the IAM fit (2026-05-28). |
| [`IAM_CONVERGENCE_GAP.md`](IAM_CONVERGENCE_GAP.md) | Investigation record: the (since-fixed) default-strategy mismatch behind an IAM cold-start convergence gap (2026-05-29). |
| [`AD_OFFSET_X3872.md`](AD_OFFSET_X3872.md) | Investigation record: AD-vs-numerical MIGRAD offset on the X(3872) dip fit — concluded expected behavior, documentation-only fix (2026-05-29). |
