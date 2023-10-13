#include "diffusion2D.h"
#include <iostream>
#include <fstream>
#include <iomanip>

Diffusion2Dsolver::Diffusion2Dsolver(const Function2D &g, const int &N): initial(g), fft(N), N(N){
    initFcoef.resize(N*N);
    double dH = 1.0 / N;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++){
            initFcoef[i*N+j] = g(dH/2+i*dH, dH/2+j*dH);
        }
    fft.apply(initFcoef, 1);
}

std::vector<double> Diffusion2Dsolver::operator () (const double &t) const{
    auto coef = initFcoef;
    for(int i = 0; i < N; i++){
        int n = (i+N/2)%N - N/2;
        for(int j = 0; j < N; j++){
            int m = (j+N/2)%N - N/2;
            coef[i*N+j] *= exp(-4*M_PI*M_PI*(n*n+m*m)*t);
        }
    }
    std::vector<double> res(N*N);
    fft.apply(coef, -1);
    for(int i = 0; i < N*N; i++)
        res[i] = coef[i].real();
    return res;
}

void Diffusion2Dsolver::output(const std::string &fname, const double &t, const double &h) const{
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::ofstream fout(fname);
    auto res = (*this)(t);
    for(const auto &x : res)
        fout << x << " ";
    fout.close();
    std::cout << "Output: Results has been saved to " << fname << std::endl;
}

void Diffusion2Dsolver::denseDiscreteOutput(const std::string &fname, const double &dt, const double &T) const{
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::ofstream fout(fname);
    fout << std::setprecision(12);
    double h = 1.0/N;
    for(double x = h/2; x <= 1; x += h)
        for(double y = h/2; y <= 1; y += h){
            fout << initial(x,y) << " ";
        }
    fout << std::endl;
    for(double t = dt; t <= T+1e-14; t += dt){
        auto res = (*this)(t);
        for(const auto &x : res)
            fout << x << " ";
        fout << std::endl;
    }
    fout.close();
    std::cout << "Dense-Discrete Output: Results has been saved to " << fname << std::endl;
}