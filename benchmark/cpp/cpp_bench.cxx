// SPDX-License-Identifier: LGPL-2.1-or-later
//
// C++ Minuit2 wall-time benchmark for Phase 0 §3.4 Criterion 2.
// Runs each FCN N times and reports median per-call wall time as JSON.

#include "Minuit2/FCNBase.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/MnContours.h"
#include "Minuit2/ContoursError.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace ROOT::Minuit2;

class Rosenbrock2 final : public FCNBase {
public:
    double operator()(const std::vector<double>& p) const override {
        return (1.0 - p[0]) * (1.0 - p[0]) + 100.0 * (p[1] - p[0] * p[0]) * (p[1] - p[0] * p[0]);
    }
    double Up() const override { return 1.0; }
};

class QuadNF final : public FCNBase {
public:
    explicit QuadNF(unsigned n) : fN(n) {}
    double operator()(const std::vector<double>& p) const override {
        double s = 0.0;
        for (unsigned i = 0; i < fN; ++i) s += p[i] * p[i];
        return s;
    }
    double Up() const override { return 1.0; }
private:
    unsigned fN;
};

class RosenbrockN final : public FCNBase {
public:
    explicit RosenbrockN(unsigned n) : fN(n) {}
    double operator()(const std::vector<double>& p) const override {
        double s = 0.0;
        for (unsigned i = 0; i + 1 < fN; ++i) {
            const double a = p[i];
            const double b = p[i + 1];
            s += 100.0 * (b - a * a) * (b - a * a) + (1.0 - a) * (1.0 - a);
        }
        return s;
    }
    double Up() const override { return 1.0; }
private:
    unsigned fN;
};

class GaussNLL final : public FCNBase {
public:
    GaussNLL(unsigned n_events, double mu, double sigma) {
        std::mt19937_64 rng(0xCAFEF00D);
        std::normal_distribution<double> g(mu, sigma);
        fData.reserve(n_events);
        for (unsigned i = 0; i < n_events; ++i) fData.push_back(g(rng));
    }
    double operator()(const std::vector<double>& p) const override {
        const double mu = p[0], sigma = p[1];
        if (sigma <= 0) return 1e30;
        const double log_sigma = std::log(sigma);
        const double sigma_squared = sigma * sigma;
        double s = 0.0;
        for (double x : fData) {
            const double d = x - mu;
            s += log_sigma + 0.5 * (d * d) / sigma_squared;
        }
        return s;
    }
    double Up() const override { return 0.5; }
private:
    std::vector<double> fData;
};

class GaussNLLNDim final : public FCNBase {
public:
    GaussNLLNDim(unsigned n_pars, unsigned n_events) : fNPars(n_pars) {
        std::mt19937_64 rng(0xCAFEF00D);
        std::normal_distribution<double> g_truth(0.0, 1.0);
        std::normal_distribution<double> g_noise(0.0, 1.0);
        fTruths.resize(n_pars);
        for (unsigned i = 0; i < n_pars; ++i) fTruths[i] = g_truth(rng);
        unsigned per = std::max(1u, n_events / n_pars);
        fData.assign(n_pars, std::vector<double>(per));
        for (unsigned i = 0; i < n_pars; ++i)
            for (unsigned j = 0; j < per; ++j)
                fData[i][j] = fTruths[i] + g_noise(rng);
    }
    double operator()(const std::vector<double>& p) const override {
        double s = 0.0;
        for (unsigned i = 0; i < fNPars; ++i) {
            const double mu = p[i];
            for (double x : fData[i]) {
                const double d = x - mu;
                s += 0.5 * d * d;
            }
        }
        return s;
    }
    double Up() const override { return 0.5; }
private:
    unsigned fNPars;
    std::vector<double> fTruths;
    std::vector<std::vector<double>> fData;
};

struct BenchResult {
    std::string name;
    double median_ns;
    int n_samples;
    double fval;
    int nfcn;
};

template <typename F>
BenchResult bench(const std::string& name, F&& make_and_run, int n_samples = 50) {
    std::vector<double> times;
    times.reserve(n_samples);
    double fval = 0.0;
    int nfcn = 0;
    for (int s = 0; s < n_samples; ++s) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = make_and_run();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
        fval = result.first;
        nfcn = result.second;
    }
    std::sort(times.begin(), times.end());
    double median = times[n_samples / 2];
    return BenchResult{name, median, n_samples, fval, nfcn};
}

int main() {
    std::vector<BenchResult> results;

    results.push_back(bench("rosenbrock_2d", []() {
        Rosenbrock2 fcn;
        MnUserParameters upar;
        upar.Add("p0", -1.2, 0.1);
        upar.Add("p1", 1.0, 0.1);
        MnMigrad migrad(fcn, upar, MnStrategy(0));
        FunctionMinimum mn = migrad();
        return std::make_pair(mn.Fval(), int(mn.NFcn()));
    }));

    results.push_back(bench("rosenbrock_10d", []() {
        RosenbrockN fcn(10);
        MnUserParameters upar;
        for (int i = 0; i < 10; ++i)
            upar.Add(("p" + std::to_string(i)).c_str(), (i % 2 == 0 ? -1.2 : 1.0), 0.1);
        MnMigrad migrad(fcn, upar, MnStrategy(0));
        FunctionMinimum mn = migrad();
        return std::make_pair(mn.Fval(), int(mn.NFcn()));
    }));

    results.push_back(bench("quad_4d", []() {
        QuadNF fcn(4);
        MnUserParameters upar;
        for (int i = 0; i < 4; ++i)
            upar.Add(("p" + std::to_string(i)).c_str(), 1.0, 0.1);
        MnMigrad migrad(fcn, upar, MnStrategy(0));
        FunctionMinimum mn = migrad();
        return std::make_pair(mn.Fval(), int(mn.NFcn()));
    }));

    results.push_back(bench("gauss_ll_2_100", []() {
        GaussNLL fcn(100, 2.0, 1.0);
        MnUserParameters upar;
        upar.Add("mu", 1.0, 0.1);
        upar.Add("sigma", 2.0, 0.1);
        MnMigrad migrad(fcn, upar, MnStrategy(0));
        FunctionMinimum mn = migrad();
        return std::make_pair(mn.Fval(), int(mn.NFcn()));
    }));

    results.push_back(bench("gauss_ll_10_1000", []() {
        GaussNLLNDim fcn(10, 1000);
        MnUserParameters upar;
        for (int i = 0; i < 10; ++i)
            upar.Add(("p" + std::to_string(i)).c_str(), 0.0, 0.1);
        MnMigrad migrad(fcn, upar, MnStrategy(0));
        FunctionMinimum mn = migrad();
        return std::make_pair(mn.Fval(), int(mn.NFcn()));
    }));

    // ── Strategy(1) variants — Phase 1 完成判据 #6 ─────────────────────
    // Mirror the Strategy(0) set but with MnStrategy(1) — iminuit's
    // default mode, where MIGRAD invokes an inner MnHesse when the
    // DFP-estimated Dcovar exceeds 0.05.

    results.push_back(bench("rosenbrock_2d_s1", []() {
        Rosenbrock2 fcn;
        MnUserParameters upar;
        upar.Add("p0", -1.2, 0.1);
        upar.Add("p1", 1.0, 0.1);
        MnMigrad migrad(fcn, upar, MnStrategy(1));
        FunctionMinimum mn = migrad();
        return std::make_pair(mn.Fval(), int(mn.NFcn()));
    }));

    results.push_back(bench("rosenbrock_10d_s1", []() {
        RosenbrockN fcn(10);
        MnUserParameters upar;
        for (int i = 0; i < 10; ++i)
            upar.Add(("p" + std::to_string(i)).c_str(), (i % 2 == 0 ? -1.2 : 1.0), 0.1);
        MnMigrad migrad(fcn, upar, MnStrategy(1));
        FunctionMinimum mn = migrad();
        return std::make_pair(mn.Fval(), int(mn.NFcn()));
    }));

    results.push_back(bench("quad_4d_s1", []() {
        QuadNF fcn(4);
        MnUserParameters upar;
        for (int i = 0; i < 4; ++i)
            upar.Add(("p" + std::to_string(i)).c_str(), 1.0, 0.1);
        MnMigrad migrad(fcn, upar, MnStrategy(1));
        FunctionMinimum mn = migrad();
        return std::make_pair(mn.Fval(), int(mn.NFcn()));
    }));

    results.push_back(bench("gauss_ll_2_100_s1", []() {
        GaussNLL fcn(100, 2.0, 1.0);
        MnUserParameters upar;
        upar.Add("mu", 1.0, 0.1);
        upar.Add("sigma", 2.0, 0.1);
        MnMigrad migrad(fcn, upar, MnStrategy(1));
        FunctionMinimum mn = migrad();
        return std::make_pair(mn.Fval(), int(mn.NFcn()));
    }));

    results.push_back(bench("gauss_ll_10_1000_s1", []() {
        GaussNLLNDim fcn(10, 1000);
        MnUserParameters upar;
        for (int i = 0; i < 10; ++i)
            upar.Add(("p" + std::to_string(i)).c_str(), 0.0, 0.1);
        MnMigrad migrad(fcn, upar, MnStrategy(1));
        FunctionMinimum mn = migrad();
        return std::make_pair(mn.Fval(), int(mn.NFcn()));
    }));

    // ── MINOS + MNCONTOUR benchmarks ──────────────────────────────
    // Each entry: (name, bench-closure). Closure runs setup + MIGRAD
    // (untimed) then the timed op (MINOS on par 0, or MNCONTOUR on
    // par 0 × par 1). Strategy(0); for these the dominant cost is the
    // inner-MIGRAD chain inside MINOS / MnContours.

    auto setup_rosen2 = []() {
        Rosenbrock2 fcn;
        MnUserParameters upar;
        upar.Add("p0", -1.2, 0.1);
        upar.Add("p1", 1.0, 0.1);
        return std::make_pair(fcn, upar);
    };
    auto setup_rosenN = [](unsigned N) {
        RosenbrockN fcn(N);
        MnUserParameters upar;
        for (unsigned i = 0; i < N; ++i)
            upar.Add(("p" + std::to_string(i)).c_str(),
                     (i % 2 == 0 ? -1.2 : 1.0), 0.1);
        return std::make_pair(fcn, upar);
    };
    auto setup_quad = [](unsigned N) {
        QuadNF fcn(N);
        MnUserParameters upar;
        for (unsigned i = 0; i < N; ++i)
            upar.Add(("p" + std::to_string(i)).c_str(), 1.0, 0.1);
        return std::make_pair(fcn, upar);
    };
    auto setup_gauss2 = []() {
        GaussNLL fcn(100, 2.0, 1.0);
        MnUserParameters upar;
        upar.Add("mu", 1.0, 0.1);
        upar.Add("sigma", 2.0, 0.1);
        return std::make_pair(fcn, upar);
    };
    auto setup_gauss10 = []() {
        GaussNLLNDim fcn(10, 1000);
        MnUserParameters upar;
        for (int i = 0; i < 10; ++i)
            upar.Add(("p" + std::to_string(i)).c_str(), 0.0, 0.1);
        return std::make_pair(fcn, upar);
    };

    // MINOS bench template — runs MIGRAD ONCE per outer sample (we
    // restore the MnUserParameterState from MIGRAD to a fresh MnMinos
    // each iteration), times only the MnMinos call.
    auto minos_bench = [](const std::string& name, auto setup_fn,
                           unsigned par_idx, int n_samples = 20) {
        std::vector<double> times;
        times.reserve(n_samples);
        double upper = 0.0;
        int nfcn_total = 0;
        for (int s = 0; s < n_samples; ++s) {
            auto [fcn, upar] = setup_fn();
            MnMigrad migrad(fcn, upar, MnStrategy(0));
            FunctionMinimum mn = migrad();
            // Re-build MnMinos from the converged minimum
            MnMinos minos(fcn, mn, MnStrategy(0));
            auto t0 = std::chrono::high_resolution_clock::now();
            MinosError me = minos.Minos(par_idx);
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            upper = me.Upper();
            nfcn_total = me.NFcn();
        }
        std::sort(times.begin(), times.end());
        return BenchResult{name, times[n_samples / 2], n_samples, upper, nfcn_total};
    };

    auto mncontour_bench = [](const std::string& name, auto setup_fn,
                               unsigned par_x, unsigned par_y,
                               unsigned npts = 30, int n_samples = 10) {
        std::vector<double> times;
        times.reserve(n_samples);
        int nfcn_total = 0;
        double last_pt = 0.0;
        for (int s = 0; s < n_samples; ++s) {
            auto [fcn, upar] = setup_fn();
            MnMigrad migrad(fcn, upar, MnStrategy(0));
            FunctionMinimum mn = migrad();
            MnContours contours(fcn, mn, MnStrategy(0));
            auto t0 = std::chrono::high_resolution_clock::now();
            ContoursError ce = contours.Contour(par_x, par_y, npts);
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            nfcn_total = ce.NFcn();
            if (!ce().empty()) last_pt = ce()[0].first;
        }
        std::sort(times.begin(), times.end());
        return BenchResult{name, times[n_samples / 2], n_samples, last_pt, nfcn_total};
    };

    // MINOS on each FCN, parameter 0
    results.push_back(minos_bench("rosenbrock_2d_minos", setup_rosen2, 0));
    results.push_back(minos_bench("rosenbrock_10d_minos",
                                   [&]{ return setup_rosenN(10); }, 0));
    results.push_back(minos_bench("quad_4d_minos",
                                   [&]{ return setup_quad(4); }, 0));
    results.push_back(minos_bench("gauss_ll_2_100_minos", setup_gauss2, 0));
    results.push_back(minos_bench("gauss_ll_10_1000_minos", setup_gauss10, 0));

    // MNCONTOUR on (par 0, par 1), 30 points
    results.push_back(mncontour_bench("rosenbrock_2d_mncontour", setup_rosen2, 0, 1));
    results.push_back(mncontour_bench("rosenbrock_10d_mncontour",
                                       [&]{ return setup_rosenN(10); }, 0, 1));
    results.push_back(mncontour_bench("quad_4d_mncontour",
                                       [&]{ return setup_quad(4); }, 0, 1));
    results.push_back(mncontour_bench("gauss_ll_2_100_mncontour", setup_gauss2, 0, 1));
    results.push_back(mncontour_bench("gauss_ll_10_1000_mncontour", setup_gauss10, 0, 1));

    // Emit JSON
    std::cout << "[\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "  {\n"
                  << "    \"name\": \"" << r.name << "\",\n"
                  << "    \"median_ns\": " << std::setprecision(17) << r.median_ns << ",\n"
                  << "    \"n_samples\": " << r.n_samples << ",\n"
                  << "    \"fval\": " << std::setprecision(17) << r.fval << ",\n"
                  << "    \"nfcn\": " << r.nfcn << "\n"
                  << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    std::cout << "]\n";
    return 0;
}
