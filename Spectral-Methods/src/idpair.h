#ifndef _IDPAIR_H_
#define _IDPAIR_H_

class idpair{
private:
    int x[2];
public:
    idpair(){
        x[0] = x[1] = 0;
    }
    idpair(const int &i, const int &j){
        x[0] = i;
        x[1] = j;
    }
    int operator [] (const int &d) const{
        return x[d];
    }
    int& operator [] (const int &d){
        return x[d];
    }
    idpair operator + (const idpair &rhs) const{
        return idpair(x[0]+rhs[0], x[1]+rhs[1]);
    }
    idpair operator - (const idpair &rhs) const{
        return idpair(x[0]-rhs[0], x[1]-rhs[1]);
    }
    friend idpair operator * (const int &k, const idpair &rhs){
        return idpair(k*rhs[0], k*rhs[1]);
    }
    idpair transDirection() const{
        return x[0] ? idpair(0,1) : idpair(1,0);
    }
};

#endif