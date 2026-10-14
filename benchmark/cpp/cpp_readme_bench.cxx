// SPDX-License-Identifier: LGPL-2.1-or-later
// Matched five-case Strategy(0) collector. See ../readme_comparison.md.
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/VariableMetricMinimizer.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnPrint.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ROOT::Minuit2;
struct Case {
    std::string name;
    unsigned kind;
    std::vector<double> x, errors;
    double up;
    const std::vector<double>* data;
};
class Objective final : public FCNBase {
public:
    explicit Objective(const Case& c) : c_(c) {}
    double Up() const override { return c_.up; }
    mutable unsigned calls=0;
    double operator()(const std::vector<double>& p) const override {
        ++calls;
        double s=0.0;
        if (c_.kind==0) { for(double x:p) s+=x*x; }
        else if (c_.kind==1) {
            return (1-p[0])*(1-p[0])+100*(p[1]-p[0]*p[0])*(p[1]-p[0]*p[0]);
        } else if (c_.kind==2) {
            for(unsigned i=0;i+1<p.size();++i) {
                double d=p[i+1]-p[i]*p[i];
                s+=100*d*d+(1-p[i])*(1-p[i]);
            }
        } else if(c_.kind==3) {
            const double mu=p[0],sigma=p[1];
            if(sigma<=0) return 1e30;
            for(double x:*c_.data) { double d=x-mu; s+=std::log(sigma)+0.5*(d*d)/(sigma*sigma); }
        } else {
            for(unsigned i=0;i<10;++i) for(unsigned j=0;j<100;++j) {
                double d=(*c_.data)[100*i+j]-p[i]; s+=0.5*d*d;
            }
        }
        return s;
    }
private:
    const Case& c_;
};
struct Output {
    double fval,edm;
    int nfcn,actual_calls;
    bool valid,has_covariance,reached_call_limit,above_max_edm;
    std::vector<double> params,errors,covariance;
};
Output operation(const Case& c) {
    Objective f(c);
    MnUserParameterState initial(c.x,c.errors);
    VariableMetricMinimizer solver;
    solver.Builder().SetStorageLevel(0);
    FunctionMinimum m=solver.Minimize(f,initial,MnStrategy(0),10000,0.1);
    Output o{m.Fval(),m.Edm(),m.NFcn(),int(f.calls),m.IsValid(),m.HasCovariance(),m.HasReachedCallLimit(),m.IsAboveMaxEdm(),{}, {}, {}};
    // Same consumer outputs as Julia: owned external parameter vector and full covariance.
    o.params.reserve(c.x.size());
    for(unsigned i=0;i<c.x.size();++i) o.params.push_back(m.State().Vec()(i));
    if(o.has_covariance) {
        const auto& cov=m.Error().InvHessian();
        o.errors.reserve(c.x.size());
        for(unsigned i=0;i<c.x.size();++i) o.errors.push_back(std::sqrt(2*c.up*cov(i,i)));
        o.covariance.reserve(c.x.size()*c.x.size());
        for(unsigned i=0;i<c.x.size();++i) for(unsigned j=0;j<c.x.size();++j) o.covariance.push_back(2*c.up*cov(i,j));
    }
    return o;
}
void number(std::ostream& o,double x) {
    if(std::isfinite(x)) o<<std::setprecision(17)<<x;
    else if(std::isnan(x)) o<<"\"NaN\"";
    else o<<(x>0?"\"Inf\"":"\"-Inf\"");
}
void array(std::ostream& o,const std::vector<double>& v) {
    o<<'[';for(unsigned i=0;i<v.size();++i){if(i)o<<',';number(o,v[i]);}o<<']';
}
void output(std::ostream& o,const Output& r) {
    o<<"{\"fval\":";number(o,r.fval);o<<",\"edm\":";number(o,r.edm);
    o<<",\"nfcn\":"<<r.nfcn<<",\"actual_calls\":"<<r.actual_calls
     <<",\"is_valid\":"<<std::boolalpha<<r.valid<<",\"has_covariance\":"<<r.has_covariance
     <<",\"reached_call_limit\":"<<r.reached_call_limit<<",\"above_max_edm\":"<<r.above_max_edm<<",\"params\":";
    array(o,r.params);o<<",\"errors\":";array(o,r.errors);o<<",\"covariance\":";array(o,r.covariance);o<<'}';
}
bool identical(const Output&a,const Output&b) {
    return a.fval==b.fval&&a.edm==b.edm&&a.nfcn==b.nfcn&&a.actual_calls==b.actual_calls&&a.valid==b.valid&&a.has_covariance==b.has_covariance&&a.reached_call_limit==b.reached_call_limit&&a.above_max_edm==b.above_max_edm&&a.params==b.params&&a.errors==b.errors&&a.covariance==b.covariance;
}
int main(int argc,char**argv) {
    try {
        if(argc!=4) throw std::runtime_error("usage: cpp_readme_bench INPUT_TRANSPORT OUTPUT_JSON SAMPLES");
        int samples=std::stoi(argv[3]);if(samples<1)throw std::runtime_error("samples must be positive");
        std::ifstream input(argv[1]);if(!input)throw std::runtime_error("input unavailable");
        std::vector<double> g2(100),g10(1000);
        for(double&x:g2)if(!(input>>x))throw std::runtime_error("invalid gauss2 input");
        for(double&x:g10)if(!(input>>x))throw std::runtime_error("invalid gauss10 input");
        std::string trailing;if(input>>trailing)throw std::runtime_error("trailing input");
        std::vector<double> r10;for(unsigned i=0;i<10;++i)r10.push_back(i%2?1.0:-1.2);
        std::vector<Case> cases={{"quad_4d",0,std::vector<double>(4,1),std::vector<double>(4,0.1),1,nullptr},
            {"rosenbrock_2d",1,{-1.2,1},std::vector<double>(2,0.1),1,nullptr},
            {"rosenbrock_10d",2,r10,std::vector<double>(10,0.1),1,nullptr},
            {"gauss_ll_2_100",3,{1,2},std::vector<double>(2,0.1),0.5,&g2},
            {"gauss_ll_10_1000",4,std::vector<double>(10,0),std::vector<double>(10,0.1),0.5,&g10}};
        MnPrint::SetGlobalLevel(-1);
        std::string final=argv[2],part=final+".tmp";
        std::ofstream out(part);if(!out)throw std::runtime_error("output unavailable");
        out<<"{\"schema_version\":1,\"language\":\"cpp\",\"clock\":\"steady_clock\",\"warmups\":5,\"evals\":1,\"rows\":[";
        for(unsigned k=0;k<cases.size();++k) {
            const auto&c=cases[k];
            Output first=operation(c);for(int w=1;w<5;++w)operation(c);
            std::vector<double>times;bool stable=true;times.reserve(samples);
            for(int s=0;s<samples;++s) {
                auto t0=std::chrono::steady_clock::now();
                Output r=operation(c);
                auto t1=std::chrono::steady_clock::now();
                times.push_back(std::chrono::duration<double,std::nano>(t1-t0).count());
                stable=stable&&identical(first,r); // Consume all fields, outside timing.
            }
            if(k)out<<',';
            out<<"{\"name\":\""<<c.name<<"\",\"samples\":"<<samples<<",\"evals\":1,\"stable_outputs\":"<<std::boolalpha<<stable<<",\"sample_times_ns\":";
            array(out,times);out<<",\"result\":";output(out,first);out<<'}';out.flush();
        }
        out<<"]}\n";out.close();if(!out)throw std::runtime_error("output write failed");
        if(std::rename(part.c_str(),final.c_str())!=0)throw std::runtime_error("output rename failed");
        return 0;
    } catch(const std::exception&e) {std::cerr<<e.what()<<'\n';return 1;}
}
