// SPDX-License-Identifier: LGPL-2.1-or-later
//
// JuMinuit C++ reference-data harness.
//
// Runs C++ Minuit2 MIGRAD on a corpus of benchmark FCNs (Rosenbrock,
// quadratic, Gaussian NLL) and dumps the FunctionMinimum + key state
// as JSON. The output is committed into test/reference_data/ and
// loaded by JuMinuit's test suite as the 1e-10 oracle.
//
// Phase 0 contract:
// - Strategy(0) — matches Julia Phase 0 lock (DR-008).
// - Numerical gradient (no analytical FCN derivative).
// - Default MachinePrecision (Float64 eps).
// - No bounds, no fixed params.
//
// Linked against the pinned Minuit2 standalone at
// reference/Minuit2_cpp/ (GooFit/Minuit2 @ 57dc936 = v6.24.0).
//
// Usage:
//   cpp_trace_harness <output_dir>
// Default output_dir is the cwd.

#include "Minuit2/FCNBase.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnUserCovariance.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnStrategy.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

using namespace ROOT::Minuit2;

// ─────────────────────────────────────────────────────────────────────
// Pinned reference metadata — must mirror docs/UPSTREAM.md.
// ─────────────────────────────────────────────────────────────────────
constexpr const char *kMinuit2Commit  = "57dc936a2b74d0b4dda1254c3dd63e7c61a97c84";
constexpr const char *kMinuit2Version = "6.24.0";

// ─────────────────────────────────────────────────────────────────────
// Benchmark FCNs
// ─────────────────────────────────────────────────────────────────────

// f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2 — the canonical 2D Rosenbrock
class Rosenbrock2 final : public FCNBase {
public:
    double operator()(const std::vector<double> &par) const override
    {
        const double a = par[0];
        const double b = par[1];
        return (1.0 - a) * (1.0 - a) + 100.0 * (b - a * a) * (b - a * a);
    }
    double Up() const override { return 1.0; }
};

// f(x) = sum_{i=0..n-1} x_i^2 — a positive-definite quadratic.
// Minimum at the origin, fval = 0. Eigenvalues of Hessian all 2.
class QuadNF final : public FCNBase {
public:
    explicit QuadNF(unsigned n) : fN(n) {}
    double operator()(const std::vector<double> &par) const override
    {
        double s = 0.0;
        for (unsigned i = 0; i < fN; ++i) s += par[i] * par[i];
        return s;
    }
    double Up() const override { return 1.0; }
private:
    unsigned fN;
};

// f(x, y) = 100 * (y - x^2)^2 + (1 - x)^2 + ... — n-dim Rosenbrock chain
// f(x) = sum_{i=0..n-2} [100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
class RosenbrockN final : public FCNBase {
public:
    explicit RosenbrockN(unsigned n) : fN(n) {}
    double operator()(const std::vector<double> &par) const override
    {
        double s = 0.0;
        for (unsigned i = 0; i + 1 < fN; ++i) {
            const double a = par[i];
            const double b = par[i + 1];
            s += 100.0 * (b - a * a) * (b - a * a) + (1.0 - a) * (1.0 - a);
        }
        return s;
    }
    double Up() const override { return 1.0; }
private:
    unsigned fN;
};

// ─────────────────────────────────────────────────────────────────────
// JSON dump
// ─────────────────────────────────────────────────────────────────────

// Print a Float64 at full precision: 17 significant digits suffice to
// round-trip an IEEE 754 double.
struct Float64Out {
    double v;
};
std::ostream &operator<<(std::ostream &os, Float64Out f)
{
    // Handle NaN/Inf explicitly — JSON has no native NaN.
    if (std::isnan(f.v)) {
        os << "\"NaN\"";
    } else if (std::isinf(f.v)) {
        os << (f.v > 0 ? "\"Inf\"" : "\"-Inf\"");
    } else {
        std::ostringstream tmp;
        tmp << std::setprecision(17) << f.v;
        os << tmp.str();
    }
    return os;
}

void dump_minimum(std::ostream &out,
                  const FunctionMinimum &mn,
                  const std::string &name,
                  const std::vector<double> &x0,
                  const std::vector<double> &errs0,
                  unsigned strategy_level)
{
    out << "{\n";
    out << "  \"name\": \"" << name << "\",\n";
    out << "  \"_meta\": {\n";
    out << "    \"source\": \"GooFit/Minuit2 @ " << kMinuit2Commit << "\",\n";
    out << "    \"version\": \"" << kMinuit2Version << "\",\n";
    out << "    \"strategy_level\": " << strategy_level << ",\n";
    out << "    \"err_def\": " << Float64Out{1.0} << ",\n";
    out << "    \"generator\": \"tools/cpp_trace_harness.cxx\"\n";
    out << "  },\n";

    // Initial conditions
    out << "  \"x0\": [";
    for (size_t i = 0; i < x0.size(); ++i) {
        if (i) out << ", ";
        out << Float64Out{x0[i]};
    }
    out << "],\n";
    out << "  \"errs0\": [";
    for (size_t i = 0; i < errs0.size(); ++i) {
        if (i) out << ", ";
        out << Float64Out{errs0[i]};
    }
    out << "],\n";

    // Final state
    out << "  \"fval\": " << Float64Out{mn.Fval()} << ",\n";
    out << "  \"edm\":  " << Float64Out{mn.Edm()} << ",\n";
    out << "  \"nfcn\": " << mn.NFcn() << ",\n";
    out << "  \"is_valid\":         " << (mn.IsValid()         ? "true" : "false") << ",\n";
    out << "  \"has_covariance\":   " << (mn.HasCovariance()   ? "true" : "false") << ",\n";
    out << "  \"has_pos_def_cov\":  " << (mn.HasPosDefCovar()  ? "true" : "false") << ",\n";
    out << "  \"hesse_failed\":     " << (mn.HesseFailed()     ? "true" : "false") << ",\n";
    out << "  \"made_pos_def\":     " << (mn.HasMadePosDefCovar() ? "true" : "false") << ",\n";
    out << "  \"reached_call_limit\": " << (mn.HasReachedCallLimit() ? "true" : "false") << ",\n";

    const MnUserParameterState &ust = mn.UserState();
    const unsigned n = static_cast<unsigned>(ust.VariableParameters());
    out << "  \"params\": [";
    for (unsigned i = 0; i < n; ++i) {
        if (i) out << ", ";
        out << Float64Out{ust.Value(i)};
    }
    out << "],\n";
    out << "  \"errors\": [";
    for (unsigned i = 0; i < n; ++i) {
        if (i) out << ", ";
        out << Float64Out{ust.Error(i)};
    }
    out << "],\n";

    // External covariance — upper triangle row-major, only authoritative
    // entries (n*(n+1)/2). Use the JuMinuit Symmetric :U convention so
    // the JSON loads directly into a parent(Symmetric(M, :U)) parent.
    if (mn.HasCovariance()) {
        const MnUserCovariance &cov = mn.UserCovariance();
        out << "  \"covariance_upper\": [";
        bool first = true;
        for (unsigned i = 0; i < n; ++i) {
            for (unsigned j = i; j < n; ++j) {
                if (!first) out << ", ";
                out << Float64Out{cov(i, j)};
                first = false;
            }
        }
        out << "]\n";
    } else {
        out << "  \"covariance_upper\": null\n";
    }
    out << "}\n";
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

struct Case {
    std::string name;
    std::vector<double> x0;
    std::vector<double> errs0;
    const FCNBase *fcn;
};

// Encode bounds + fixed flags for a parameter.
struct ParamMeta {
    bool has_lower = false;
    bool has_upper = false;
    double lower = 0.0;
    double upper = 0.0;
    bool fixed = false;
};

template <typename Fn>
void run_case(const std::string &outdir,
              const std::string &name,
              const std::vector<double> &x0,
              const std::vector<double> &errs0,
              const Fn &fcn,
              unsigned strategy_level = 0,
              const std::vector<ParamMeta> *meta = nullptr)
{
    MnUserParameters upar;
    for (size_t i = 0; i < x0.size(); ++i) {
        upar.Add("p" + std::to_string(i), x0[i], errs0[i]);
        if (meta && i < meta->size()) {
            const auto &m = (*meta)[i];
            if (m.has_lower && m.has_upper) {
                upar.SetLimits(i, m.lower, m.upper);
            } else if (m.has_upper) {
                upar.SetUpperLimit(i, m.upper);
            } else if (m.has_lower) {
                upar.SetLowerLimit(i, m.lower);
            }
            if (m.fixed) {
                upar.Fix(i);
            }
        }
    }
    MnStrategy stra(strategy_level);
    MnMigrad migrad(fcn, upar, stra);
    FunctionMinimum mn = migrad();

    const std::string path = outdir + "/" + name + ".json";
    std::ofstream out(path);
    if (!out) {
        std::cerr << "ERROR: could not open " << path << "\n";
        std::exit(2);
    }
    dump_minimum(out, mn, name, x0, errs0, strategy_level);
    std::cout << "[ok] " << path
              << "  fval=" << std::setprecision(17) << mn.Fval()
              << "  nfcn=" << mn.NFcn()
              << "  valid=" << (mn.IsValid() ? "y" : "n") << "\n";
}

// MnMinos oracle: run MIGRAD then MINOS on all free params, dump
// asymmetric ± errors per parameter. Phase 1 完成判据 #5 (1e-8 agreement
// with C++ MINOS on at least one bounded and one unbounded fit).
void run_minos_case(const std::string &outdir,
                    const std::string &name,
                    const std::vector<double> &x0,
                    const std::vector<double> &errs0,
                    const FCNBase &fcn,
                    unsigned strategy_level = 1)
{
    MnUserParameters upar;
    for (size_t i = 0; i < x0.size(); ++i) {
        upar.Add("p" + std::to_string(i), x0[i], errs0[i]);
    }
    MnStrategy stra(strategy_level);
    MnMigrad migrad(fcn, upar, stra);
    FunctionMinimum mn = migrad();

    if (!mn.IsValid()) {
        std::cerr << "WARN: " << name << " MIGRAD invalid; skipping MINOS\n";
        return;
    }

    MnMinos minos(fcn, mn, stra);

    const std::string path = outdir + "/" + name + "_minos.json";
    std::ofstream out(path);
    if (!out) {
        std::cerr << "ERROR: could not open " << path << "\n";
        std::exit(2);
    }

    const unsigned n = static_cast<unsigned>(mn.UserState().VariableParameters());
    out << "{\n";
    out << "  \"name\": \"" << name << "\",\n";
    out << "  \"_meta\": {\n";
    out << "    \"source\": \"GooFit/Minuit2 @ " << kMinuit2Commit << "\",\n";
    out << "    \"version\": \"" << kMinuit2Version << "\",\n";
    out << "    \"strategy_level\": " << strategy_level << ",\n";
    out << "    \"err_def\": " << Float64Out{1.0} << ",\n";
    out << "    \"generator\": \"tools/cpp_trace_harness.cxx :: run_minos_case\"\n";
    out << "  },\n";
    out << "  \"fval\": " << Float64Out{mn.Fval()} << ",\n";
    out << "  \"params\": [";
    for (unsigned i = 0; i < n; ++i) {
        if (i) out << ", ";
        out << Float64Out{mn.UserState().Value(i)};
    }
    out << "],\n";

    out << "  \"minos\": [\n";
    for (unsigned i = 0; i < n; ++i) {
        MinosError me = minos.Minos(i);
        out << "    {\n";
        out << "      \"par\": " << i << ",\n";
        out << "      \"min_value\": " << Float64Out{me.Min()} << ",\n";
        out << "      \"upper\": " << Float64Out{me.Upper()} << ",\n";
        out << "      \"lower\": " << Float64Out{me.Lower()} << ",\n";
        out << "      \"upper_valid\": " << (me.UpperValid() ? "true" : "false") << ",\n";
        out << "      \"lower_valid\": " << (me.LowerValid() ? "true" : "false") << ",\n";
        out << "      \"upper_new_min\": " << (me.AtUpperMaxFcn() ? "false" : (me.AtUpperLimit() ? "false" : "false")) << ",\n";
        out << "      \"lower_new_min\": " << (me.AtLowerMaxFcn() ? "false" : (me.AtLowerLimit() ? "false" : "false")) << ",\n";
        out << "      \"nfcn\": " << me.NFcn() << "\n";
        out << "    }" << (i + 1 < n ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    std::cout << "[ok] " << path
              << "  fval=" << std::setprecision(17) << mn.Fval()
              << "  npar=" << n << "\n";
}

int main(int argc, char *argv[])
{
    const std::string outdir = (argc > 1) ? argv[1] : ".";

    std::cout << "JuMinuit C++ reference-data harness — Minuit2 "
              << kMinuit2Version << " @ " << kMinuit2Commit << "\n";
    std::cout << "Output directory: " << outdir << "\n";

    Rosenbrock2 rosen2;
    run_case(outdir, "rosenbrock_2d", { -1.2, 1.0 }, { 0.1, 0.1 }, rosen2);

    QuadNF quad4(4);
    run_case(outdir, "quad_4d",
             { 1.0, 1.0, 1.0, 1.0 },
             { 0.1, 0.1, 0.1, 0.1 },
             quad4);

    RosenbrockN rosen10(10);
    run_case(outdir, "rosenbrock_10d",
             { -1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0 },
             { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 },
             rosen10);

    // ── Bounded test cases (Phase 1 §3.4 Criterion 1 oracle) ──
    // bounded_quad_2d: f = (x - 0.5)² + (y - 0.5)² with x ∈ [0, 1]
    QuadNF quad2(2);
    {
        std::vector<ParamMeta> meta(2);
        meta[0].has_lower = true; meta[0].lower = 0.0;
        meta[0].has_upper = true; meta[0].upper = 1.0;
        // Note: harness FCN is sum(x_i²); to test bounded fit at the
        // interior optimum we shift in the bounded case via subclass.
        // Use a direct lambda-wrapper here:
    }
    struct CenteredQuad : public FCNBase {
        explicit CenteredQuad(double cx, double cy) : fCx(cx), fCy(cy) {}
        double operator()(const std::vector<double> &p) const override {
            return (p[0] - fCx) * (p[0] - fCx) + (p[1] - fCy) * (p[1] - fCy);
        }
        double Up() const override { return 1.0; }
    private:
        double fCx, fCy;
    };
    CenteredQuad cq(0.5, 0.5);
    {
        std::vector<ParamMeta> meta(2);
        meta[0].has_lower = true; meta[0].lower = 0.0;
        meta[0].has_upper = true; meta[0].upper = 1.0;
        run_case(outdir, "bounded_sin_2d",
                 { 0.3, 0.3 }, { 0.1, 0.1 }, cq, 0, &meta);
    }
    // bounded_lower_2d: lower-only bound; minimum well inside the bound
    CenteredQuad cq2(3.0, 2.0);
    {
        std::vector<ParamMeta> meta(2);
        meta[0].has_lower = true; meta[0].lower = 1.0;
        run_case(outdir, "bounded_lower_2d",
                 { 2.0, 1.0 }, { 0.1, 0.1 }, cq2, 0, &meta);
    }
    // bounded_upper_2d: upper-only bound
    CenteredQuad cq3(-1.0, 0.0);
    {
        std::vector<ParamMeta> meta(2);
        meta[0].has_upper = true; meta[0].upper = 5.0;
        run_case(outdir, "bounded_upper_2d",
                 { 0.0, 1.0 }, { 0.1, 0.1 }, cq3, 0, &meta);
    }
    // quad_2d_fixed: y fixed at 5; x free; minimum of f at x=1, fval=(5-2)²=9
    struct Shifted2D : public FCNBase {
        double operator()(const std::vector<double> &p) const override {
            return (p[0] - 1.0) * (p[0] - 1.0) + (p[1] - 2.0) * (p[1] - 2.0);
        }
        double Up() const override { return 1.0; }
    };
    Shifted2D sh;
    {
        std::vector<ParamMeta> meta(2);
        meta[1].fixed = true;
        run_case(outdir, "quad_2d_fixed_y",
                 { 0.0, 5.0 }, { 0.1, 0.1 }, sh, 0, &meta);
    }

    // ── MINOS oracles (Phase 1 完成判据 #5) ───────────────────────────
    // 1e-8 agreement check on at least one unbounded fit. C++ MnMinos
    // uses Strategy(1) by default in iminuit; we mirror that here.
    {
        Rosenbrock2 r2;
        run_minos_case(outdir, "rosenbrock_2d", { -1.2, 1.0 }, { 0.1, 0.1 }, r2);
    }
    {
        QuadNF q4(4);
        run_minos_case(outdir, "quad_4d",
                       { 1.0, 1.0, 1.0, 1.0 }, { 0.1, 0.1, 0.1, 0.1 }, q4);
    }
    {
        Shifted2D sh2;
        run_minos_case(outdir, "quad_2d_shifted",
                       { 0.0, 0.0 }, { 0.1, 0.1 }, sh2);
    }

    return 0;
}
