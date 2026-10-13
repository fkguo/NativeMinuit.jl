// SPDX-License-Identifier: LGPL-2.1-or-later
// Reference for the NativeMinuit orchestration, not a Minuit2 iterate API.
#include "Minuit2/FCNBase.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnSimplex.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnUserParameterState.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
using namespace ROOT::Minuit2;

struct Objective : FCNBase {
  enum Kind { Rugged, Smooth, Singular } kind;
  mutable unsigned calls=0;
  explicit Objective(Kind k) : kind(k) {}
  double operator()(const std::vector<double>& x) const override {
    ++calls;
    if (kind == Smooth) return (x[0]-1)*(x[0]-1)+(x[1]-2)*(x[1]-2)+10;
    if (kind == Singular) return x[0]*x[0];
    return .3*(x[0]*x[0]+x[1]*x[1])
      +1e-3*(std::abs(std::sin(1e3*x[0]))+std::abs(std::sin(1e3*x[1])))
      -8*std::exp(-((x[0]-.8)*(x[0]-.8)+(x[1]-.8)*(x[1]-.8))/.3);
  }
  double Up() const override { return 1; }
};

// Matches the public wrapper's unbounded error conversion and positive-finite
// resume fallback. C++ UserState alone has a different invalid-error fallback.
MnUserParameters resume(const FunctionMinimum& m, const std::vector<double>& base) {
  MnUserParameters p;
  for (unsigned i=0;i<base.size();++i) {
    double e=base[i];
    if (m.Error().IsAvailable()) e=std::sqrt(std::max(2*m.Up()*m.Error().InvHessian()(i,i),0.));
    if (!std::isfinite(e) || e<=0) e=base[i];
    p.Add(i==0 ? "x" : "y",m.Parameters().Vec()(i),e);
  }
  return p;
}

bool better(const FunctionMinimum& cand, const FunctionMinimum& best) {
  if (!std::isfinite(cand.Fval())) return false;
  if (!std::isfinite(best.Fval())) return true;
  if (cand.IsValid()!=best.IsValid()) return cand.IsValid();
  return cand.Fval()<best.Fval();
}
using Visit=std::pair<std::vector<double>,double>;
Visit visit(const FunctionMinimum& m) {
  return {{m.Parameters().Vec()(0),m.Parameters().Vec()(1)},m.Fval()};
}
bool revisit(const FunctionMinimum& m,const std::vector<Visit>& seen,const std::vector<double>& base) {
  const auto v=visit(m);
  if (!std::isfinite(v.second)) return false;
  for (const auto& old:seen) {
    if (!std::isfinite(old.second) || std::abs(v.second-old.second)>std::max(1e-12,1e-6*std::abs(old.second))) continue;
    bool same=true;
    for (unsigned i=0;i<base.size();++i) {
      const double scale=std::max({std::abs(v.first[i]),std::abs(old.first[i]),std::abs(base[i]),1e-12});
      if (std::abs(v.first[i]-old.first[i])>1e-2*scale) {same=false;break;}
    }
    if (same) return true;
  }
  return false;
}

struct Result { FunctionMinimum best; int passes,calls,bestpass; };
void trace(const std::string& name,int iteration,bool sx,int pass,const FunctionMinimum& m) {
  std::cerr<<name<<" iterate="<<iteration<<" simplex="<<sx<<" pass="<<pass
    <<" valid="<<m.IsValid()<<" limit="<<m.HasReachedCallLimit()<<" calls="<<m.NFcn()
    <<" x="<<m.Parameters().Vec()(0)<<" y="<<m.Parameters().Vec()(1)
    <<" fval="<<m.Fval()<<" edm="<<m.Edm()<<" seed_edm="<<m.Seed().State().Edm()<<'\n';
  const auto& s=m.Seed().State();
  for (unsigned i=0;i<2;++i) std::cerr<<"seed coordinate="<<i<<" x="<<s.Vec()(i)
    <<" grad="<<s.Gradient().Grad()(i)<<" g2="<<s.Gradient().G2()(i)
    <<" gstep="<<s.Gradient().Gstep()(i)<<" inverse_diag="<<s.Error().InvHessian()(i,i)<<'\n';
}
Result run(const Objective& f,const std::string& name,const std::vector<double>& x,
           const std::vector<double>& base,int iteration,bool sx,unsigned maxfcn,double tol) {
  const auto before=f.calls;
  auto current=MnMigrad(f,x,base,1)(maxfcn,tol);
  auto best=current;
  int passes=1,bestpass=1;
  std::vector<Visit> seen{visit(current)};
  trace(name,iteration,sx,passes,current);
  for (int pass=2;pass<=iteration;++pass) {
    if (current.IsValid() || current.HasReachedCallLimit()) break;
    auto p=resume(current,base);
    std::cerr<<"resume "<<name<<" simplex="<<sx<<" pass="<<pass<<" x="<<p.Value(0)
      <<" y="<<p.Value(1)<<" error_x="<<p.Error(0)<<" error_y="<<p.Error(1)
      <<" prior_covariance="<<(!sx && current.Error().IsAvailable() && current.Error().IsValid())<<'\n';
    if (sx) {
      // NativeMinuit carries values and simplex extent errors, no covariance.
      auto simplex=MnSimplex(f,p,2)(maxfcn,tol);
      p=simplex.UserParameters();
      for (unsigned i=0;i<base.size();++i)
        if (!std::isfinite(p.Error(i)) || p.Error(i)<=0) p.SetError(i,base[i]);
      std::cerr<<"simplex "<<name<<" pass="<<pass<<" x="<<p.Value(0)<<" y="<<p.Value(1)
        <<" error_x="<<p.Error(0)<<" error_y="<<p.Error(1)<<" calls="<<simplex.NFcn()<<'\n';
      current=MnMigrad(f,p,2)(maxfcn,tol);
    } else if (current.Error().IsAvailable() && current.Error().IsValid()) {
      current=MnMigrad(f,p,current.UserCovariance(),2)(maxfcn,tol);
    } else {
      current=MnMigrad(f,p,2)(maxfcn,tol);
    }
    ++passes;
    trace(name,iteration,sx,passes,current);
    if (better(current,best)) {best=current;bestpass=pass;}
    if (revisit(current,seen,base)) break;
    seen.push_back(visit(current));
  }
  return {best,passes,static_cast<int>(f.calls-before),bestpass};
}
void row(const std::string& name,int iteration,bool sx,const Result& r) {
  const auto& m=r.best;
  const bool edm_is_fval=m.States().size()==1 && m.Edm()==m.Fval() && m.Edm()!=m.Seed().State().Edm();
  std::cout<<name<<' '<<iteration<<' '<<sx<<' '<<m.IsValid()<<' '<<m.HasReachedCallLimit()
    <<' '<<r.passes<<' '<<r.calls<<' '<<r.bestpass<<' '<<m.Parameters().Vec()(0)<<' '<<m.Parameters().Vec()(1)
    <<' '<<m.Fval()<<' '<<m.Edm()<<' '<<edm_is_fval<<' '<<m.Error().IsValid();
  for (auto ij:std::vector<std::pair<unsigned,unsigned>>{{0,0},{0,1},{1,1}})
    std::cout<<' '<<(m.Error().IsAvailable() ? 2*m.Up()*m.Error().InvHessian()(ij.first,ij.second) : NAN);
  std::cout<<'\n';
}
int main(int argc,char** argv) {
  std::cout<<std::setprecision(17);std::cerr<<std::setprecision(17);
  Objective rugged(Objective::Rugged),smooth(Objective::Smooth),singular(Objective::Singular);
  if (argc==2 && (std::string(argv[1])=="--shared-seed" || std::string(argv[1])=="--shared-trace")) {
    if (std::string(argv[1])=="--shared-trace") MnPrint::SetGlobalLevel(3);
    // Diagnostic: both languages receive these identical literal values and
    // errors. This does not replace the repeated-policy comparison above.
    MnUserParameters p;
    p.Add("x",0.79477858309322069,0.017406206117379787);
    p.Add("y",0.79442618648243957,0.050000000000000003);
    const auto before=rugged.calls;
    auto sx=MnSimplex(rugged,p,2)(4000,1e-6);
    row("shared_simplex",1,true,{sx,1,static_cast<int>(rugged.calls-before),1});
    p=sx.UserParameters();
    const auto mid=rugged.calls;
    auto mg=MnMigrad(rugged,p,2)(4000,1e-6);
    row("shared_s2",1,true,{mg,1,static_cast<int>(rugged.calls-mid),1});
    trace("shared_s2",1,true,1,mg);
    return 0;
  }
  for (bool sx:{false,true}) for (int iteration:{1,5}) {
    row("rugged",iteration,sx,run(rugged,"rugged",{.8,.8},{.05,.05},iteration,sx,4000,1e-6));
    row("smooth",iteration,sx,run(smooth,"smooth",{0,0},{.1,.1},iteration,sx,4000,1e-6));
  }
  auto valid=run(smooth,"selector_valid",{0,0},{.1,.1},1,false,4000,1e-6);
  auto invalid=run(singular,"selector_invalid",{.1,.1},{.05,.05},1,false,4000,1e-6);
  row("selector_valid",1,false,valid);row("selector_invalid",1,false,invalid);
  // Independent control rejects cost-only selection and verifies order symmetry.
  std::cout<<"selector "<<better(valid.best,invalid.best)<<' '<<better(invalid.best,valid.best)
    <<' '<<valid.best.IsValid()<<' '<<invalid.best.IsValid()<<' '<<(valid.best.Fval()>invalid.best.Fval())<<'\n';
}
