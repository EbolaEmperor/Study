#include "advdif2D.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>

//------------------------------------------------------Settings----------------------------------------------------------------

AdvectionDiffusionSolver::AdvectionDiffusionSolver(const int &M): M(M), dH(1.0/M), difSolver(M){
    dT = 0.0;
    f = nullptr;
    initial = nullptr;
    ux = nullptr;
    uy = nullptr;
    uIsConst = false;
}

void AdvectionDiffusionSolver::setInitial(Function2D *_initial){
    initial = _initial;
}

void AdvectionDiffusionSolver::setVelocity(Function2D *_ux, Function2D *_uy){
    ux = _ux;
    uy = _uy;
    uIsConst = false;
}

void AdvectionDiffusionSolver::setConstVelocity(const double &_ux, const double &_uy){
    constUx = _ux;
    constUy = _uy;
    uIsConst = true;
}

void AdvectionDiffusionSolver::setEndTime(const double &_tEnd){
    tEnd = _tEnd;
}

void AdvectionDiffusionSolver::setNu(const double &_nu){
    difSolver.setDiffusionCoef(_nu);
}

void AdvectionDiffusionSolver::setTimeStepWithCaurant(const double &caurant, const double &maxux, const double &maxuy){
    if(dH==0.0){
        std::cerr << "[Error] setTimeStepWithCaurant: Please set grid size first." << std::endl;
    } else {
        dT = caurant / (maxux/dH + maxuy/dH);
    }
}

//---------------------------------------------------Discrete Operators---------------------------------------------------------

double AdvectionDiffusionSolver::facephi_right(const ColVector &phi, const int &i, const int &j){
    return 7.0/12.0 * (solValue(phi,i,j) + solValue(phi,i+1,j)) - 1.0/12.0 * (solValue(phi,i-1,j) + solValue(phi,i+2,j));
}

double AdvectionDiffusionSolver::facephi_up(const ColVector &phi, const int &i, const int &j){
    return 7.0/12.0 * (solValue(phi,i,j) + solValue(phi,i,j+1)) - 1.0/12.0 * (solValue(phi,i,j-1) + solValue(phi,i,j+2));
}

double AdvectionDiffusionSolver::Gdp_phi_right(const ColVector &phi, const int &i, const int &j){
    return 0.5/dH * (facephi_right(phi,i,j+1) - facephi_right(phi,i,j-1));
}

double AdvectionDiffusionSolver::Gdp_phi_up(const ColVector &phi, const int &i, const int &j){
    return 0.5/dH * (facephi_up(phi,i+1,j) - facephi_up(phi,i-1,j));
}

double AdvectionDiffusionSolver::Gdp_u_right(Function2D *u, const int &i, const int &j){
    return 0.5/dH * (u->intFixX((i+1)*dH,(j+1)*dH,(j+2)*dH) - u->intFixX((i+1)*dH,(j-1)*dH,j*dH)) /dH;
}

double AdvectionDiffusionSolver::Gdp_u_up(Function2D *u, const int &i, const int &j){
    return 0.5/dH * (u->intFixY((j+1)*dH,(i+1)*dH,(i+2)*dH) - u->intFixY((j+1)*dH,(i-1)*dH,i*dH)) /dH;
}

double AdvectionDiffusionSolver::F_right(const ColVector &phi, Function2D *u, const int &i, const int &j){
    return facephi_right(phi,i,j) * u->intFixX((i+1)*dH,j*dH,(j+1)*dH)/dH + dH*dH/12.0 * Gdp_phi_right(phi,i,j) * Gdp_u_right(u,i,j);
}

double AdvectionDiffusionSolver::F_up(const ColVector &phi, Function2D *u, const int &i, const int &j){
    return facephi_up(phi,i,j) * u->intFixY((j+1)*dH,i*dH,(i+1)*dH)/dH + dH*dH/12.0 * Gdp_phi_up(phi,i,j) * Gdp_u_up(u,i,j);
}

ColVector AdvectionDiffusionSolver::Ladv(const ColVector &phi){
    ColVector res(M*M);
    if(uIsConst){
        for(int i = 0; i < M; i++)
            for(int j = 0; j < M; j++){
                res(idx(i,j)) = -constUx/dH * ( 2.0/3.0*solValue(phi,i+1,j) - 2.0/3.0*solValue(phi,i-1,j) - 1.0/12.0*solValue(phi,i+2,j) + 1.0/12.0*solValue(phi,i-2,j) )
                                -constUy/dH * ( 2.0/3.0*solValue(phi,i,j+1) - 2.0/3.0*solValue(phi,i,j-1) - 1.0/12.0*solValue(phi,i,j+2) + 1.0/12.0*solValue(phi,i,j-2) );
            }
    } else {
        for(int i = 0; i < M; i++)
            for(int j = 0; j < M; j++){
                res(idx(i,j)) = -1.0/dH * ( F_right(phi,ux,i,j) - F_right(phi,ux,i-1,j) + F_up(phi,uy,i,j) - F_up(phi,uy,i,j-1) );
            }
    }
    return res;
}

//---------------------------------------------------Sol Value and Index--------------------------------------------------------

int AdvectionDiffusionSolver::idx(const int &i, const int &j){
    return i * M + j;
}

int AdvectionDiffusionSolver::idx(const idpair &x){
    return x[0] * M + x[1];
}

double AdvectionDiffusionSolver::solValue(const ColVector &phi, const int &i, const int &j){
    return phi(idx( (i+M)%M, (j+M)%M ));
}

//---------------------------------------------------output and check error---------------------------------------------------------

void AdvectionDiffusionSolver::output(const std::string &outname){
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::ofstream out(outname);
    out << std::fixed << std::setprecision(16);
    out << sol.reshape(M,M) << std::endl;
    out.close();
    std::cout << "Result has been saved to " << outname << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
}

//-----------------------------------------------------Solve Process----------------------------------------------------------

void AdvectionDiffusionSolver::solve(){
    std::cout << "Setting initial values..." << std::endl;
    sol = ColVector(M*M);
    for(int i = 0; i < M; i++)
        for(int j = 0; j < M; j++){
            sol(idx(i,j)) = initial->accInt2D(i*dH, (i+1)*dH, j*dH, (j+1)*dH) * M * M;
        }
    int stcl = clock();
    for(double t = 0.0; t+1e-12 < tEnd; t += dT){
        // multi stage
        std::cout << "Time: " << t << std::endl;

        // FV-SE alternating base on strang spliting
        auto midsol = sol + dT/4 * Ladv(sol);
        sol = sol + dT/2 * Ladv(midsol);

        difSolver.init(sol);
        sol = difSolver(dT);

        midsol = sol + dT/4 * Ladv(sol);
        sol = sol + dT/2 * Ladv(midsol);
    }
    std::cout << "Solved. in " << (double)(clock()-stcl)/CLOCKS_PER_SEC << "s." << std::endl;
}