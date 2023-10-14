#ifndef _FUNC_2D_
#define _FUNC_2D_

class Function2D{
public:
    virtual double operator () (const double &x, const double &y) const = 0;
    double at (const double &x, const double &y) const;
    virtual double intFixX(const double &x, const double &d, const double &u) const;
    virtual double intFixY(const double &y, const double &d, const double &u) const;
    virtual double int2D(const double &l, const double &r, const double &d, const double &u) const;
    virtual double _accInt2D(const double &l, const double &r, const double &d, const double &u, const double &A) const;
    virtual double accInt2D(const double &l, const double &r, const double &d, const double &u) const;
};

#endif