#ifndef _FFT_2D_
#define _FFT_2D_

#define onFFTW true

#include <vector>
#include <complex>
#include <fftw3.h>

using Complex = std::complex<double>;
using Array = std::vector<Complex>;
// Using an n*n 1D array to stoarge an (n,n) 2D array.

class fft2D{
private:
    // The size of array.
    int N;

    // The array for butterfly transformation.
    std::vector<int> r;

    // The pre-processed unit complex roots
    std::vector<Array> wns, iwns;

    // Arrays for fftw
    fftw_complex *in;
    fftw_plan p, pinv;

    // Use FFTW if set to be true [default=false].
    bool useFFTW;

    // Initialize the unit roots or initialize FFTW.
    void init();

    // Apply fft2D with FFTW to a. v=1 for fft and v=-1 for ifft.
    void applyFFTW(Array &a, const int &v) const;

public:
    ~fft2D();
    fft2D(const int &_N);
    fft2D(const int &_N, const bool &p);

    // Apply fft2D to a. v=1 for fft and v=-1 for ifft.
    void apply(Array &a, const int &v) const;
};

#endif