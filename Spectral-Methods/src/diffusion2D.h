#ifndef _DIFFUSION_2D_
#define _DIFFUSION_2D_

#include <Eigen/Sparse>
#include "fft2D.h"
#include "function2D.h"

class Diffusion2Dsolver{
private:
    int N;
    double nu;
    fft2D fft;
    Array initFcoef;

public:
    // Initialize fft2D.
    Diffusion2Dsolver(const int &N);

    // Set diffusion coefficient, default is 1.0.
    void setDiffusionCoef(const double &nu);

    // Initialize with a function.
    void init(const Function2D &initial);

    // Initialize with discrete values.
    void init(const Eigen::VectorXd &initial);

    // Return the solution at time t.
    Eigen::VectorXd operator () (const double &t) const;

    // Output the solution at time t.
    void output(const std::string &fname, const double &t, const double &h) const;

    // Output the solutions at time 0, dt, 2dt, ..., T/
    void denseDiscreteOutput(const std::string &fname, const double &dt, const double &T) const;
};

#endif