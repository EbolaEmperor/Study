#include "fft2D.h"
#include <bits/stdc++.h>
using namespace std;

const int N = 64;
double f[N][N];
double g[N][N];
double h[N<<1][N<<1];

void generateCoef(){
    srand(time(0));
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++){
            f[i][j] = (double)rand()/RAND_MAX;
            g[i][j] = (double)rand()/RAND_MAX;
        }
}

void testInv(){
    fft2D fft(N);
    vector<Complex> a(N*N, 0);
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++){
            a[i*N+j] = f[i][j];
        }
    fft.apply(a, 1);
    fft.apply(a, -1);
    double maxerr = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
           maxerr = max(maxerr, fabs(a[i*N+j].real()-f[i][j]));
    }
    cout << "fft-ifft max-error: " << maxerr << endl;
}

void naiveConv(){
    memset(h, 0, sizeof(h));
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            for(int k = 0; k < N; k++)
                for(int l = 0; l < N; l++)
                    h[i+k][j+l] += f[i][j] * g[k][l];
}

void fftConv(){
    fft2D fft(N*2);
    vector<Complex> a(N*N*4, 0), b(N*N*4, 0);
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++){
            a[i*2*N+j] = f[i][j];
            b[i*2*N+j] = g[i][j];
        }
    fft.apply(a, 1);
    fft.apply(b, 1);
    for(int i = 0; i < N*N*4; i++)
        a[i] *= b[i];
    fft.apply(a, -1);
    double maxerr = 0;
    for(int i = 0; i < N*2; i++){
        for(int j = 0; j < N*2; j++){
            maxerr = max(maxerr, fabs(a[i*N*2+j].real()-h[i][j]));
        }
    }
    cout << "convolution max-error: " << maxerr << endl;
}

int main(){
    cout << "test size: " << N << endl;
    generateCoef();
    testInv();
    naiveConv();
    fftConv();
    return 0;
}