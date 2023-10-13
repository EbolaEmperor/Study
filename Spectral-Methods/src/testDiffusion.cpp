#include "diffusion2D.h"
#include <bits/stdc++.h>
using namespace std;

const int N = 256;

class G : public Function2D{
public:
    virtual double operator () (const double &x, const double &y) const{
        static const double rcx = 0.5;
        static const double rcy = 0.5;
        return exp( (-(x-rcx)*(x-rcx)-(y-rcy)*(y-rcy)) * 60 );
    }
} g;

int main(){
    Diffusion2Dsolver solver(g, N);
    solver.denseDiscreteOutput("result-dense.txt", 0.01, 0.3);
    return 0;
}