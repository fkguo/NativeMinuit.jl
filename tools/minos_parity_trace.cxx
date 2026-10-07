// SPDX-License-Identifier: LGPL-2.1-or-later
//
// MINOS parity reference generator (C++ Minuit2 6.24, the pinned reference).
//
// Runs MIGRAD + HESSE + MINOS (all at one explicit strategy, tol 0.1) on
// five generic problems whose data live in test/reference_data/
// minos_parity_*.txt, wrapping each objective in a counting FCN that also
// records the scanned parameter's value on every call during MINOS. The
// per-probe (value, calls) trace, the MINOS end points, the call counts and
// the crossing quality q = (F_profile - F_min)/up at each end point are
// written as JSON to stdout; the committed copy is
// test/reference_data/minos_parity_cpp.json, consumed by
// test/test_minos_crossing_parity.jl.
//
// Build (same way as benchmark/cpp, against the pinned standalone):
//   cmake -S tools -B tools/build -DCMAKE_BUILD_TYPE=Release
//   cmake --build tools/build --target minos_parity_trace
//   tools/build/minos_parity_trace test/reference_data 1 > test/reference_data/minos_parity_cpp.json
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnPrint.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace ROOT::Minuit2;

struct Counted final : public FCNBase {
    std::function<double(const std::vector<double>&)> f;
    double up;
    mutable long ncalls = 0;
    mutable int scan = -1;
    mutable std::vector<double> trace;
    Counted(std::function<double(const std::vector<double>&)> f_, double up_) : f(f_), up(up_) {}
    double operator()(const std::vector<double>& p) const override {
        ++ncalls;
        if (scan >= 0) trace.push_back(p[scan]);
        return f(p);
    }
    double Up() const override { return up; }
};

struct Problem {
    std::string name;
    std::function<double(const std::vector<double>&)> f;
    double up;
    std::vector<double> x0, err0;
    std::vector<int> lower_limited; // indices with lower limit 0
};

static std::vector<std::vector<double>> read_rows(const std::string& path) {
    std::ifstream in(path);
    std::vector<std::vector<double>> rows; std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line); std::vector<double> r; double v;
        while (ss >> v) r.push_back(v);
        if (!r.empty()) rows.push_back(r);
    }
    return rows;
}

static std::vector<std::pair<double,long>> collapse(const std::vector<double>& t) {
    std::vector<std::pair<double,long>> out;
    for (double v : t) {
        if (!out.empty() && out.back().first == v) out.back().second++;
        else out.push_back({v, 1});
    }
    return out;
}

static std::string g17(double v) { char b[64]; std::snprintf(b, 64, "%.17g", v); return b; }

int main(int argc, char** argv) {
    std::string datadir = argc > 1 ? argv[1] : "test/reference_data";
    int strategy = argc > 2 ? std::atoi(argv[2]) : 1;
    double tol = 0.1;
    auto gauss = read_rows(datadir + "/minos_parity_gauss5.txt");
    auto bw = read_rows(datadir + "/minos_parity_bw6.txt");
    auto ex = read_rows(datadir + "/minos_parity_exp2.txt");
    auto Hr = read_rows(datadir + "/minos_parity_quad4_H.txt");

    std::vector<Problem> probs;
    probs.push_back({"rosen2", [](const std::vector<double>& p){ return (1-p[0])*(1-p[0]) + 100*(p[1]-p[0]*p[0])*(p[1]-p[0]*p[0]); },
                     1.0, {-1.2, 1.0}, {0.1, 0.1}, {}});
    probs.push_back({"quad4", [Hr](const std::vector<double>& p){ double s=0; for(int i=0;i<4;++i) for(int j=0;j<4;++j) s += 0.5*p[i]*Hr[i][j]*p[j]; return s; },
                     1.0, {1,1,1,1}, {0.1,0.1,0.1,0.1}, {}});
    probs.push_back({"gauss5", [gauss](const std::vector<double>& p){ double s=0; for (auto& r : gauss){ int k=(int)r[0]; double mu=p[2*k], sg=p[2*k+1]; if (sg<=0) return 1e30; double d=r[1]-mu; s += std::log(sg) + 0.5*d*d/(sg*sg);} return s; },
                     0.5, {1.3,1.0,2.3,1.0,3.3,1.0,4.3,1.0,5.3,1.0}, std::vector<double>(10,0.1), {}});
    probs.push_back({"bw6", [bw](const std::vector<double>& p){ double s=0; for (auto& r : bw){ double x=r[0]; double m = p[0]*(p[2]*p[2]/4)/((x-p[1])*(x-p[1]) + p[2]*p[2]/4) + p[3] + p[4]*(x-5) + p[5]*(x-5)*(x-5); double d=(r[1]-m)/0.05; s+=d*d;} return s; },
                     1.0, {2.5, 5.1, 1.0, 0.9, 0.0, 0.0}, std::vector<double>(6,0.1), {2}});
    probs.push_back({"exp2", [ex](const std::vector<double>& p){ double s=0; for (auto& r : ex){ double m = p[0]*std::exp(-r[0]/p[1]); double d=(r[1]-m)/0.02; s+=d*d;} return s; },
                     1.0, {1.5, 1.0}, {0.1, 0.1}, {}});

    std::cout << "{\n \"impl\": \"cpp_minuit2_6.24\", \"strategy\": " << strategy << ", \"tol\": " << tol << ",\n \"problems\": {\n";
    bool firstp = true;
    for (auto& P : probs) {
        Counted fcn(P.f, P.up);
        MnUserParameters upar;
        for (size_t i = 0; i < P.x0.size(); ++i) upar.Add("p" + std::to_string(i), P.x0[i], P.err0[i]);
        for (int i : P.lower_limited) upar.SetLowerLimit(i, 0.0);
        MnStrategy stra(strategy);
        MnMigrad migrad(fcn, upar, stra);
        FunctionMinimum fm = migrad(0, tol);
        long n_migrad = fcn.ncalls;
        MnHesse hesse(stra);
        hesse(fcn, fm, 0);
        long n_hesse = fcn.ncalls - n_migrad;
        unsigned n = fm.UserState().VariableParameters();
        if (!firstp) std::cout << ",\n"; firstp = false;
        std::cout << "  \"" << P.name << "\": {\n   \"valid\": " << (fm.IsValid()?"true":"false")
                  << ", \"fval\": " << g17(fm.Fval()) << ", \"edm\": " << g17(fm.Edm())
                  << ", \"nfcn_migrad\": " << n_migrad << ", \"nfcn_hesse\": " << n_hesse << ",\n   \"values\": [";
        for (unsigned i = 0; i < n; ++i) std::cout << (i?", ":"") << g17(fm.UserState().Value(i));
        std::cout << "],\n   \"errors\": [";
        for (unsigned i = 0; i < n; ++i) std::cout << (i?", ":"") << g17(fm.UserState().Error(i));
        std::cout << "],\n   \"minos\": [\n";
        MnMinos minos(fcn, fm, stra);
        for (unsigned i = 0; i < n; ++i) {
            fcn.scan = (int)i; fcn.trace.clear();
            long c0 = fcn.ncalls;
            MinosError me = minos.Minos(i, 0, tol);
            long dc = fcn.ncalls - c0;
            fcn.scan = -1;
            auto probes = collapse(fcn.trace);
            // crossing quality q: re-minimise the others with p_i fixed at the end point
            auto q_of = [&](double v){
                MnUserParameters u2;
                for (unsigned j = 0; j < n; ++j) u2.Add("p"+std::to_string(j), fm.UserState().Value(j), fm.UserState().Error(j));
                for (int k : P.lower_limited) u2.SetLowerLimit(k, 0.0);
                u2.SetValue(i, v); u2.Fix(i);
                MnMigrad mg2(fcn, u2, MnStrategy(2));
                FunctionMinimum f2 = mg2(0, 1e-4);
                return (f2.Fval() - fm.Fval()) / P.up;
            };
            double xmin = me.Min();
            double q_up = q_of(xmin + me.Upper()), q_lo = q_of(xmin + me.Lower());
            std::cout << "    {\"par\": " << i << ", \"min\": " << g17(xmin) << ", \"lower\": " << g17(me.Lower()) << ", \"upper\": " << g17(me.Upper())
                      << ", \"lower_valid\": " << (me.LowerValid()?"true":"false") << ", \"upper_valid\": " << (me.UpperValid()?"true":"false")
                      << ", \"at_lower_limit\": " << (me.AtLowerLimit()?"true":"false") << ", \"at_upper_limit\": " << (me.AtUpperLimit()?"true":"false")
                      << ", \"nfcn\": " << me.NFcn() << ", \"nfcn_counted\": " << dc
                      << ", \"q_lower\": " << g17(q_lo) << ", \"q_upper\": " << g17(q_up) << ",\n     \"probes\": [";
            for (size_t k = 0; k < probes.size(); ++k) std::cout << (k?", ":"") << "[" << g17(probes[k].first) << ", " << probes[k].second << "]";
            std::cout << "]}" << (i+1<n?",":"") << "\n";
        }
        std::cout << "   ]\n  }";
    }
    std::cout << "\n }\n}\n";
    return 0;
}
