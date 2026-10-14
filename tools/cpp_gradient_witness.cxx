// SPDX-License-Identifier: LGPL-2.1-or-later
// Bounded gradient and final quadratic reference; stdout is the fixture table.
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnUserFcn.h"
#include "Minuit2/InitialGradientCalculator.h"
#include "Minuit2/Numerical2PGradientCalculator.h"
#include "Minuit2/MinimumParameters.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnStrategy.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
using namespace ROOT::Minuit2;
class Quad: public FCNBase { public: double center; explicit Quad(double c):center(c){} double operator()(const std::vector<double>&x)const override {return (x[0]-center)*(x[0]-center);} double Up()const override{return 1.;}};
int main(){std::cout<<std::setprecision(17);struct Case{const char*name;int kind;double x,e,center;};
for(auto c:std::vector<Case>{{"lower_large",1,3,100,1},{"upper_large",2,1,100,2},{"both_large",3,1,100,2},{"near_lower",1,1e-7,1e-5,1},{"near_upper",2,4-1e-7,1e-5,2},{"near_both",3,1e-7,1e-5,2},{"lower_small",1,3,.1,1},{"upper_small",2,1,.1,2},{"both_small",3,1,.1,2}}){
Quad q(c.center);MnUserParameters p;p.Add("x",c.x,c.e);if(c.kind==1)p.SetLowerLimit(0,0);if(c.kind==2)p.SetUpperLimit(0,4);if(c.kind==3)p.SetLimits(0,0,4);
MnUserParameterState s(p);MnStrategy st(0);MnUserFcn f(q,s.Trafo());MnAlgebraicVector x(1);x(0)=s.IntParameters()[0];MinimumParameters pa(x,f(x));
InitialGradientCalculator ig(f,s.Trafo(),st);auto g=ig(pa);Numerical2PGradientCalculator ng(f,s.Trafo(),st);auto gn=ng(pa,g);
MnAlgebraicVector gg(1),g2(1),gs(1);gg(0)=1;g2(0)=1e-14;gs(0)=10;FunctionGradient previous(gg,g2,gs);auto forced=ng(pa,previous);
auto m=MnMigrad(q,p,st)(1000,1e-12);
std::cout<<c.name<<' '<<x(0)<<' '<<g.Grad()(0)<<' '<<g.G2()(0)<<' '<<g.Gstep()(0)<<' '<<gn.Grad()(0)<<' '<<gn.G2()(0)<<' '<<gn.Gstep()(0)<<' '<<forced.Grad()(0)<<' '<<forced.G2()(0)<<' '<<forced.Gstep()(0)<<' '<<m.IsValid()<<' '<<m.UserState().Value(0)<<' '<<m.Fval()<<' '<<m.Edm()<<' '<<m.UserCovariance()(0,0)<<'\n';
}}
