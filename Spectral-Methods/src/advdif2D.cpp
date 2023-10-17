#include "advdif2D.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <queue>

//------------------------------------------------------Settings----------------------------------------------------------------

AdvectionDiffusionSolver::AdvectionDiffusionSolver(const int &M): 
    M(M), 
    dH(1.0/M), 
    difSolver(M), 
    LadvOp(M*M,M*M), 
    sol(M*M),
    u_face0(M*M),
    u_face1(M*M),
    Gdpu_face0(M*M),
    Gdpu_face1(M*M)
{
    dT = 0.0;
    f = nullptr;
    initial = nullptr;
    ux = nullptr;
    uy = nullptr;
}

void AdvectionDiffusionSolver::setInitial(Function2D *_initial){
    initial = _initial;
}

void AdvectionDiffusionSolver::setVelocity(Function2D *_ux, Function2D *_uy){
    ux = _ux;
    uy = _uy;
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

void AdvectionDiffusionSolver::facephi_right(const int &i, const int &j, const double &coef){
    aRaw[idx(i,j)] += 7.0/12.0*coef;
    aRaw[idx(i+1,j)] += 7.0/12.0*coef;
    aRaw[idx(i-1,j)] -= 1.0/12.0*coef;
    aRaw[idx(i+2,j)] -= 1.0/12.0*coef;
}

void AdvectionDiffusionSolver::facephi_up(const int &i, const int &j, const double &coef){
    aRaw[idx(i,j)] += 7.0/12.0*coef;
    aRaw[idx(i,j+1)] += 7.0/12.0*coef;
    aRaw[idx(i,j-1)] -= 1.0/12.0*coef;
    aRaw[idx(i,j+2)] -= 1.0/12.0*coef;
}

void AdvectionDiffusionSolver::Gdp_phi_right(const int &i, const int &j, const double &coef){
    facephi_right(i,j+1, 0.5/dH*coef);
    facephi_right(i,j-1, -0.5/dH*coef);
}

void AdvectionDiffusionSolver::Gdp_phi_up(const int &i, const int &j, const double &coef){
    facephi_up(i+1,j, 0.5/dH*coef);
    facephi_up(i-1,j, -0.5/dH*coef);
}

void AdvectionDiffusionSolver::F_right(const int &i, const int &j, const double &coef){
    facephi_right(i,j, solValue(u_face0,i,j)*coef);
    Gdp_phi_right(i,j, dH*dH/12.0*solValue(Gdpu_face0,i,j)*coef);
}

void AdvectionDiffusionSolver::F_up(const int &i, const int &j, const double &coef){
    facephi_up(i,j, solValue(u_face1,i,j)*coef);
    Gdp_phi_up(i,j, dH*dH/12.0*solValue(Gdpu_face1,i,j)*coef);
}

void AdvectionDiffusionSolver::constructLadv(){
    std::vector< Eigen::Triplet<double> > eles;
    for(int i = 0; i < M; i++)
        for(int j = 0; j < M; j++){
            F_right(i, j, -1.0/dH);
            F_right(i-1, j, 1.0/dH);
            F_up(i, j, -1.0/dH);
            F_up(i, j-1, 1.0/dH);
            int idx = i*M+j;
            for(auto& p : aRaw){
                if(fabs(p.second) < 1e-15) continue;
                eles.emplace_back(idx, p.first, p.second);
            }
            aRaw.clear();
        }
    LadvOp.setFromTriplets(eles.begin(), eles.end());
    eles.clear();
}

Eigen::VectorXd AdvectionDiffusionSolver::Ladv(const Eigen::VectorXd &phi){
    return LadvOp * phi;
}

//---------------------------------------------------Sol Value and Index--------------------------------------------------------

int AdvectionDiffusionSolver::idx(const int &i, const int &j){
    return (i+M)%M * M + (j+M)%M;
}

inline double AdvectionDiffusionSolver::solValue(const Eigen::VectorXd &phi, const int &i, const int &j){
    return phi[idx(i,j)];
}

//---------------------------------------------------output and check error---------------------------------------------------------

void AdvectionDiffusionSolver::output(const std::string &outname){
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::ofstream out(outname);
    out << std::fixed << std::setprecision(16);
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++){
            out << sol[i*M+j] << " ";
        }
        out << std::endl;
    }
    out.close();
    std::cout << "Result has been saved to " << outname << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
}

//-----------------------------------------------------Solve Process----------------------------------------------------------

void AdvectionDiffusionSolver::AdvectionStep(const double &t){
    // Advection step based on classical RK.
    CPUTimer timer;
    auto RK_y1 = Ladv(sol);
    auto RK_y2 = Ladv(sol + t/2*RK_y1);
    auto RK_y3 = Ladv(sol + t/2*RK_y2);
    auto RK_y4 = Ladv(sol + t*RK_y3);
    sol += t/6 * (RK_y1 + 2*RK_y2 + 2*RK_y3 + RK_y4);
    advTime += timer();
}

void AdvectionDiffusionSolver::DiffusionStep(const double &t){
    // Diffusion step based on spectral method.
    CPUTimer timer;
    difSolver.init(sol);
    sol = difSolver(t);
    difTime += timer();
}

void AdvectionDiffusionSolver::StrangStep(const double &t){
    // FV-SE alternating base on strang splitting.
    AdvectionStep(t/2);
    DiffusionStep(t);
    AdvectionStep(t/2);
}

void AdvectionDiffusionSolver::solve(){
    std::cout << "Setting initial values and constructing Ladv..." << std::endl;
    for(int i = 0; i < M; i++)
        for(int j = 0; j < M; j++){
            sol(idx(i,j)) = initial->accInt2D(i*dH, (i+1)*dH, j*dH, (j+1)*dH) * M * M;
            u_face0(idx(i,j)) = ux->intFixX((i+1)*dH, j*dH, (j+1)*dH)/dH;
            u_face1(idx(i,j)) = uy->intFixY((j+1)*dH, i*dH, (i+1)*dH)/dH;
        }
    for(int i = 0; i < M; i++)
        for(int j = 0; j < M; j++){
            Gdpu_face0(idx(i,j)) = 0.5/dH * (solValue(u_face0,i,j+1) - solValue(u_face0,i,j-1));
            Gdpu_face1(idx(i,j)) = 0.5/dH * (solValue(u_face1,i+1,j) - solValue(u_face1,i-1,j));
        }
    constructLadv();
    
    // The coefficients for Forest-Ruth splitting.
    // const double w1 = 1.0 / (2.0-pow(2.0,1.0/3));
    // const double w2 = -pow(2.0,1.0/3) / (2.0-pow(2.0,1.0/3));

    int stcl = clock();
    for(double t = 0.0; t+1e-12 < tEnd; t += dT){
        std::cout << "Time: " << t << std::endl;
        // Forest-Ruth splitting.
        // StrangStep(w1*dT);
        // StrangStep(w2*dT);
        // StrangStep(w1*dT);

        // Chin splitting.
        AdvectionStep(dT/6);
        DiffusionStep(dT/2);
        AdvectionStep(dT*2/3);
        DiffusionStep(dT/2);
        AdvectionStep(dT/6);
    }
    std::cout << "Solved. in " << (double)(clock()-stcl)/CLOCKS_PER_SEC << "s." << std::endl;
    std::cout << "Advection-step time: " << advTime << "s." << std::endl;
    std::cout << "Diffusion-step time: " << difTime << "s." << std::endl;
}