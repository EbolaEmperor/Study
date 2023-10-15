#include "advdif2D.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <queue>

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

double AdvectionDiffusionSolver::Gdp_phi_right(const ColVector &phi_face0, const int &i, const int &j){
    return 0.5/dH * (solValue(phi_face0,i,j+1) - solValue(phi_face0,i,j-1));
}

double AdvectionDiffusionSolver::Gdp_phi_up(const ColVector &phi_face1, const int &i, const int &j){
    return 0.5/dH * (solValue(phi_face1,i+1,j) - solValue(phi_face1,i-1,j));
}

double AdvectionDiffusionSolver::F_right(const ColVector &phi, Function2D *u, const int &i, const int &j){
    return solValue(phi_face0,i,j) * solValue(u_face0,i,j) + dH*dH/12.0 * Gdp_phi_right(phi_face0,i,j) * Gdp_phi_right(u_face0,i,j);
}

double AdvectionDiffusionSolver::F_up(const ColVector &phi, Function2D *u, const int &i, const int &j){
    return solValue(phi_face1,i,j) * solValue(u_face1,i,j) + dH*dH/12.0 * Gdp_phi_up(phi_face1,i,j) * Gdp_phi_up(u_face1,i,j);
}

ColVector AdvectionDiffusionSolver::Ladv(const ColVector &phi){
    for(int i = 0; i < M; i++)
        for(int j = 0; j < M; j++){
            phi_face0(idx(i,j)) = facephi_right(phi, i, j);
            phi_face1(idx(i,j)) = facephi_up(phi, i, j);
            u_face0(idx(i,j)) = ux->intFixX((i+1)*dH, j*dH, (j+1)*dH)/dH;
            u_face1(idx(i,j)) = uy->intFixY((j+1)*dH, i*dH, (i+1)*dH)/dH;
        }
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
                F_face0(idx(i,j)) = F_right(phi,ux,i,j);
                F_face1(idx(i,j)) = F_up(phi,ux,i,j);
            }
        for(int i = 0; i < M; i++)
            for(int j = 0; j < M; j++){
                res(idx(i,j)) = -1.0/dH * ( solValue(F_face0,i,j) - solValue(F_face0,i-1,j) + solValue(F_face1,i,j) - solValue(F_face1,i,j-1) );
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

void AdvectionDiffusionSolver::AdvectionStep(const double &t){
    // Advection step based on classical RK.
    ColVector RK_y1 = Ladv(sol);
    ColVector RK_y2 = Ladv(sol + t/2*RK_y1);
    ColVector RK_y3 = Ladv(sol + t/2*RK_y2);
    ColVector RK_y4 = Ladv(sol + t*RK_y3);
    sol = sol + t/6 * (RK_y1 + 2*RK_y2 + 2*RK_y3 + RK_y4);
}

void AdvectionDiffusionSolver::DiffusionStep(const double &t){
    // Diffusion step based on spectral method.
    difSolver.init(sol);
    sol = difSolver(t);
}

void AdvectionDiffusionSolver::StrangStep(const double &t){
    // FV-SE alternating base on strang splitting.
    AdvectionStep(t/2);
    DiffusionStep(t);
    AdvectionStep(t/2);
}

void AdvectionDiffusionSolver::solve(){
    std::cout << "Setting initial values..." << std::endl;
    sol = ColVector(M*M);
    for(int i = 0; i < M; i++)
        for(int j = 0; j < M; j++){
            sol(idx(i,j)) = initial->accInt2D(i*dH, (i+1)*dH, j*dH, (j+1)*dH) * M * M;
        }
    phi_face0 = ColVector(M*M);
    phi_face1 = ColVector(M*M);
    u_face0 = ColVector(M*M);
    u_face1 = ColVector(M*M);
    F_face0 = ColVector(M*M);
    F_face1 = ColVector(M*M);
    const double w1 = 1.0 / (2.0-pow(2.0,1.0/3));
    const double w2 = -pow(2.0,1.0/3) / (2.0-pow(2.0,1.0/3));

    int stcl = clock();
    for(double t = 0.0; t+1e-12 < tEnd; t += dT){
        std::cout << "Time: " << t << std::endl;
        // Forest-Ruth splitting.
        StrangStep(w1*dT);
        StrangStep(w2*dT);
        StrangStep(w1*dT);
    }
    std::cout << "Solved. in " << (double)(clock()-stcl)/CLOCKS_PER_SEC << "s." << std::endl;
}