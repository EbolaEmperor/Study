#include "diffusion2D.h"
#include <bits/stdc++.h>
using namespace std;

class G : public Function2D{
public:
    virtual double operator () (const double &x, const double &y) const{
        static const double rcx = 0.5;
        static const double rcy = 0.5;
        return exp( (-(x-rcx)*(x-rcx)-(y-rcy)*(y-rcy)) * 60 );
    }
} g;

int main(int argc, char * argv[]){
    Diffusion2Dsolver solver(stoi(argv[1]));
    solver.init(g);
    solver.denseDiscreteOutput("result-dense.txt", 0.01, 0.3);
    return 0;
}