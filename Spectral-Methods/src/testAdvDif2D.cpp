#include <bits/stdc++.h>
#include "advdif2D.h"
using namespace std;

const double nu = 0.001;

class INITPHI : public Function2D{
public:
    double operator () (const double &x, const double &y) const{
        static const double regulizer = 100 * log(1e16);
        static const double rcx = 0.5;
        static const double rcy = 0.75;
        return exp( (-(x-rcx)*(x-rcx)-(y-rcy)*(y-rcy)) * regulizer );
    }
} initphi;

class FUNCUX : public Function2D{
public:
    double operator () (const double &x, const double &y) const{
        return 0.1 * sin(M_PI*x) * sin(M_PI*x) * sin(2*M_PI*y);
    }
    double intFixX(const double &x, const double &d, const double &u) const{
        return 0.1 * sin(M_PI*x) * sin(M_PI*x) * (cos(2*M_PI*d) - cos(2*M_PI*u)) / (2*M_PI);
    }
    double intFixY(const double &y, const double &d, const double &u) const{
        return 0.1 * sin(2*M_PI*y) * (2*M_PI*(u-d) + sin(2*M_PI*d) - sin(2*M_PI*u)) / (4*M_PI);
    }
} ux;

class FUNCUY : public Function2D{
public:
    double operator () (const double &x, const double &y) const{
        return -0.1 * sin(2*M_PI*x) * sin(M_PI*y) * sin(M_PI*y);
    }
    double intFixX(const double &x, const double &d, const double &u) const{
        return -0.1 * sin(2*M_PI*x) * (2*M_PI*(u-d) + sin(2*M_PI*d) - sin(2*M_PI*u)) / (4*M_PI);
    }
    double intFixY(const double &y, const double &d, const double &u) const{
        return -0.1 * sin(M_PI*y) * sin(M_PI*y) * (cos(2*M_PI*d) - cos(2*M_PI*u)) / (2*M_PI);
    }
} uy;

int main(int argc, char * argv[]){
    AdvectionDiffusionSolver solver(stoi(argv[1]));
    solver.setEndTime(10.0);
    solver.setNu(nu);
    solver.setTimeStepWithCaurant(1.0, 0.1, 0.1);
    solver.setInitial(&initphi);
    solver.setVelocity(&ux, &uy);
    solver.solve();
    solver.output("result.txt");
    return 0;
}