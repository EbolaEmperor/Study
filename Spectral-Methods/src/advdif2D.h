#ifndef _ADV_DIF_H_
#define _ADV_DIF_H_

// An advection-diffusion solver based on FV-SE alternating method.

#include "idpair.h"
#include "function2D.h"
#include "diffusion2D.h"
#include <cstring>
#include <unordered_map>
#include <Eigen/Sparse>

class AdvectionDiffusionSolver{
private:
    int M;
    double tEnd;
    double dH, dT;
    Function2D *f, *initial, *ux, *uy;
    // ux: velocity in x;    uy: velocity in y
    Eigen::VectorXd sol;
    Diffusion2Dsolver difSolver;
    Eigen::VectorXd u_face0, u_face1;
    Eigen::VectorXd Gdpu_face0, Gdpu_face1;
    std::unordered_map<int, double> aRaw;
    Eigen::SparseMatrix<double> LadvOp;
    double advTime, difTime;

    // Discrete operators
    Eigen::VectorXd Ladv(const Eigen::VectorXd &phi);
    void F_up(const int &i, const int &j, const double &coef);
    void F_right(const int &i, const int &j, const double &coef);
    void Gdp_phi_up(const int &i, const int &j, const double &coef);
    void Gdp_phi_right(const int &i, const int &j, const double &coef);
    void facephi_up(const int &i, const int &j, const double &coef);
    void facephi_right(const int &i, const int &j, const double &coef);
    void constructLadv();

    // map the 2D index into 1D index
    int idx(const int &i, const int &j);
    double solValue(const Eigen::VectorXd &phi, const int &i, const int &j);
    void AdvectionStep(const double &t);
    void DiffusionStep(const double &t);
    void StrangStep(const double &t);

public:
    AdvectionDiffusionSolver(const int &M);
    void solve();
    void setInitial(Function2D *_initial);
    void setVelocity(Function2D *_ux, Function2D *_uy);
    void setEndTime(const double &_tEnd);
    void setNu(const double &_nu);
    void setTimeStepWithCaurant(const double &caurant, const double &maxux, const double &maxuy);
    void output(const std::string &outname);
};

#endif