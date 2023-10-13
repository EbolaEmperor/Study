#ifndef _DIFFUSION_2D_
#define _DIFFUSION_2D_

#include "fft2D.h"

class Function2D{
public:
    virtual double operator () (const double &x, const double &y) const = 0;
};

class Diffusion2Dsolver{
private:
    int N;
    const Function2D &initial;
    fft2D fft;
    Array initFcoef;

public:
    // Initialize fft2D and do the fft to the initial condition.
    Diffusion2Dsolver(const Function2D &g, const int &N);

    // Return the solution at time t.
    std::vector<double> operator () (const double &t) const;

    // Output the solution at time t.
    void output(const std::string &fname, const double &t, const double &h) const;

    // Output the solutions at time 0, dt, 2dt, ..., T/
    void denseDiscreteOutput(const std::string &fname, const double &dt, const double &T) const;
};

#endif