#include "fft2D.h"
#include <cstring>

fft2D::~fft2D(){
    for(auto &arr : wns) Array().swap(arr);
    for(auto &arr : iwns) Array().swap(arr);
    std::vector<Array>().swap(wns);
    std::vector<Array>().swap(iwns);
    std::vector<int>().swap(r);
    N = 0;
}

fft2D::fft2D(const int &_N){
    N = _N;
    r.resize(N,0);
    int l = 0;
    for(int i = 1; i < N; i <<= 1) l++;
    wns.resize(l);
    iwns.resize(l);
	for(int i = 0; i < N; i++)
        r[i] = ( r[i/2] / 2 ) | ( (i&1) << (l-1) );
	for(int i = 1, l = 0; i < N; i <<= 1, l++){
		Complex wn(cos(M_PI/i), sin(M_PI/i));
        Complex iwn(cos(M_PI/i), -sin(M_PI/i));
		int p = i << 1;
        Complex w(1,0), iw(1,0);
        wns[l].resize(i);
        iwns[l].resize(i);
        for(int k1 = 0; k1 < i; k1++){
            wns[l][k1] = w;
            iwns[l][k1] = iw;
            w = w * wn;
            iw = iw * iwn;
        }
	}
}

void fft2D::apply(Array &a, const int &v) const{
	for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            if(i<r[i] || i==r[i] && j<r[j]) swap(a[i*N+j], a[r[i]*N+r[j]]);
	for(int i = 1, l = 0; i < N; i <<= 1, l++){
		int p = i << 1;
        const auto& wns_t = (v==1) ? wns[l] : iwns[l];
		for(int j1 = 0; j1 < N; j1 += p)
            for(int j2 = 0; j2 < N; j2 += p)
            {
                for(int k1 = 0; k1 < i; k1++){
                    const auto& w1 = wns_t[k1];
                    for(int k2 = 0; k2 < i; k2++)
                    {
                        const auto& w2 = wns_t[k2];
                        Complex A1 = a[(j1+k1)*N+(j2+k2)];
                        Complex A2 = w2*a[(j1+k1)*N+(j2+k2+i)];
                        Complex A3 = w1*a[(j1+k1+i)*N+(j2+k2)];
                        Complex A4 = w1*w2*a[(j1+k1+i)*N+(j2+k2+i)];
                        a[(j1+k1)*N+(j2+k2)]     = A1 + A2 + A3 + A4;
                        a[(j1+k1)*N+(j2+k2+i)]   = A1 - A2 + A3 - A4;
                        a[(j1+k1+i)*N+(j2+k2)]   = A1 + A2 - A3 - A4;
                        a[(j1+k1+i)*N+(j2+k2+i)] = A1 - A2 - A3 + A4;
                    }
                }
            }
	}
    if(v==-1) for(auto& z : a) z/=(N*N);
}