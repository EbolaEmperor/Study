#ifndef _FFT_2D_
#define _FFT_2D_

#include <vector>
#include <complex>

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

public:
    ~fft2D();

    // Initialize the butterfly transformation and unit complex roots.
    fft2D(const int &_N);

    // Apply fft2D to a. v=1 for fft and v=-1 for ifft.
    void apply(Array &a, const int &v) const;
};

#endif