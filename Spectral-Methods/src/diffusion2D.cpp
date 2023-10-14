#include "diffusion2D.h"
#include <iostream>
#include <fstream>
#include <iomanip>

Diffusion2Dsolver::Diffusion2Dsolver(const int &N): fft(N), N(N), nu(1.0){}

void Diffusion2Dsolver::setDiffusionCoef(const double &_nu){
    nu = _nu;
}

void Diffusion2Dsolver::init(const Function2D &initial){
    initFcoef.resize(N*N);
    double h = 1.0/N;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++){
            initFcoef[i*N+j] = initial.accInt2D(i*h, (i+1)*h, j*h, (j+1)*h) / (h*h);
        }
    fft.apply(initFcoef, 1);
}

void Diffusion2Dsolver::init(const ColVector &initial){
    initFcoef.resize(N*N);
    for(int i = 0; i < N*N; i++)
        initFcoef[i] = initial(i);
    fft.apply(initFcoef, 1);
}

ColVector Diffusion2Dsolver::operator () (const double &t) const{
    auto coef = initFcoef;
    for(int i = 0; i < N; i++){
        int n = (i+N/2)%N - N/2;
        for(int j = 0; j < N; j++){
            int m = (j+N/2)%N - N/2;
            coef[i*N+j] *= exp(-4*M_PI*M_PI*nu*(n*n+m*m)*t);
        }
    }
    ColVector res(N*N);
    fft.apply(coef, -1);
    for(int i = 0; i < N*N; i++)
        res(i) = coef[i].real();
    return res;
}

void Diffusion2Dsolver::output(const std::string &fname, const double &t, const double &h) const{
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::ofstream fout(fname);
    auto res = (*this)(t);
    for(int i = 0; i < res.size(); i++)
        fout << res(i) << " ";
    fout.close();
    std::cout << "Output: Results has been saved to " << fname << std::endl;
}

void Diffusion2Dsolver::denseDiscreteOutput(const std::string &fname, const double &dt, const double &T) const{
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::ofstream fout(fname);
    fout << std::setprecision(12);
    double h = 1.0/N;
    for(double t = dt; t <= T+1e-14; t += dt){
        auto res = (*this)(t);
        for(int i = 0; i < res.size(); i++)
            fout << res(i) << " ";
        fout << std::endl;
    }
    fout.close();
    std::cout << "Dense-Discrete Output: Results has been saved to " << fname << std::endl;
}