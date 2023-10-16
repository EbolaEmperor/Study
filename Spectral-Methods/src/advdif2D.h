#ifndef _ADV_DIF_H_
#define _ADV_DIF_H_

// An advection-diffusion solver based on FV-SE alternating method.

#include "matrix.h"
#include "idpair.h"
#include "function2D.h"
#include "diffusion2D.h"
#include <cstring>

class AdvectionDiffusionSolver{
private:
    int M;
    double tEnd;
    double dH, dT;
    Function2D *f, *initial, *ux, *uy;
    // ux: velocity in x;    uy: velocity in y
    bool uIsConst;
    double constUx, constUy;
    ColVector sol;
    Diffusion2Dsolver difSolver;
    ColVector phi_face0, phi_face1;
    ColVector u_face0, u_face1;
    ColVector Gdpu_face0, Gdpu_face1;
    ColVector F_face0, F_face1;
    double advTime, difTime;

    // Discrete operators
    ColVector Ladv(const ColVector &phi);
    double F_up(const int &i, const int &j);
    double F_right(const int &i, const int &j);
    double Gdp_phi_up(const ColVector &phi, const int &i, const int &j);
    double Gdp_phi_right(const ColVector &phi, const int &i, const int &j);
    double facephi_up(const ColVector &phi, const int &i, const int &j);
    double facephi_right(const ColVector &phi, const int &i, const int &j);

    // map the 2D index into 1D index
    int idx(const int &i, const int &j);
    int idx(const idpair &x);
    double solValue(const ColVector &phi, const int &i, const int &j);
    void AdvectionStep(const double &t);
    void DiffusionStep(const double &t);
    void StrangStep(const double &t);

public:
    AdvectionDiffusionSolver(const int &M);
    void solve();
    void setInitial(Function2D *_initial);
    void setVelocity(Function2D *_ux, Function2D *_uy);
    void setConstVelocity(const double &_ux, const double &_uy);
    void setEndTime(const double &_tEnd);
    void setNu(const double &_nu);
    void setTimeStepWithCaurant(const double &caurant, const double &maxux, const double &maxuy);
    void output(const std::string &outname);
};

#endif