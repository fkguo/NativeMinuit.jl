// SPDX-License-Identifier: LGPL-2.1-or-later
//
// C++ Minuit2 wall-time benchmark for Phase 0 §3.4 Criterion 2.
// Runs each FCN N times and reports median per-call wall time as JSON.

#include "Minuit2/FCNBase.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnStrategy.h"

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
        double s = 0.0;
        for (double x : fData) {
            const double d = x - mu;
            s += std::log(sigma) + 0.5 * (d * d) / (sigma * sigma);
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
