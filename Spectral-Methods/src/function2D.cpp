#include "function2D.h"
#include <cmath>

inline double Function2D::at (const double &x, const double &y) const{
    return (*this)(x,y);
}

double Function2D::intFixX(const double &x, const double &d, const double &u) const{
    return (u-d)/6.0 * (at(x,d) + 4.0 * at(x,(d+u)/2.0) + at(x,u));
}

double Function2D::intFixY(const double &y, const double &d, const double &u) const{
    return (u-d)/6.0 * (at(d,y) + 4.0 * at((d+u)/2.0,y) + at(u,y));
}

double Function2D::int2D(const double &l, const double &r, const double &d, const double &u) const{
    static const double simpsonCoef[3][3] = {
        1, 4, 1,
        4, 16, 4,
        1, 4, 1
    };
    const double dx = (r-l) / 2;
    const double dy = (u-d) / 2;
    double res = 0;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++){
            res += simpsonCoef[i][j] * at(l+i*dx, d+j*dy);
        }
    return res * dx * dy / 9;
}

double Function2D::_accInt2D(const double &l, const double &r, const double &d, const double &u, const double &A) const{
    const double midx = (l+r)/2;
    const double midy = (d+u)/2;
    const double A1 = int2D(l,midx,d,midy);
    const double A2 = int2D(midx,r,d,midy);
    const double A3 = int2D(l,midx,midy,u);
    const double A4 = int2D(midx,r,midy,u);
    return fabs(A1+A2+A3+A4-A)<1e-14 ? A1+A2+A3+A4 : 
        _accInt2D(l,midx,d,midy,A1)+_accInt2D(midx,r,d,midy,A2)+
        _accInt2D(l,midx,midy,u,A3)+_accInt2D(midx,r,midy,u,A4);

}

double Function2D::accInt2D(const double &l, const double &r, const double &d, const double &u) const{
    return _accInt2D(l, r, d, u, int2D(l,r,d,u));
}
