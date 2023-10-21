CPP=icpx
LIB=-lfftw3
OPT=-O2 -O3 -Ofast
SO=/home/wenchong/intel/oneapi/mkl/2023.2.0/lib/intel64/libmkl_rt.so

all: testAdvDif2D testDiffusion testFFT

testAdvDif2D: fft2D.o diffusion2D.o function2D.o advdif2D.o testAdvDif2D.o
	$(CPP) fft2D.o diffusion2D.o function2D.o advdif2D.o testAdvDif2D.o $(SO) -o testAdvDif2D $(OPT) $(LIB)

testDiffusion: fft2D.o diffusion2D.o testDiffusion.o function2D.o
	$(CPP) fft2D.o diffusion2D.o testDiffusion.o function2D.o -o testDiffusion $(OPT) $(LIB)

testFFT: fft2D.o testFFT.o
	$(CPP) testFFT.o fft2D.o -o testFFT $(OPT) $(LIB)

testAdvDif2D.o: testAdvDif2D.cpp
	$(CPP) -c testAdvDif2D.cpp $(OPT)

testDiffusion.o: testDiffusion.cpp
	$(CPP) -c testDiffusion.cpp $(OPT)

testFFT.o: testFFT.cpp
	$(CPP) -c testFFT.cpp $(OPT)

fft2D.o: fft2D.h fft2D.cpp
	$(CPP) -c fft2D.cpp $(OPT)

diffusion2D.o: diffusion2D.h diffusion2D.cpp fft2D.h function2D.h
	$(CPP) -c diffusion2D.cpp $(OPT)

advdif2D.o: advdif2D.cpp advdif2D.h diffusion2D.h fft2D.h function2D.h
	$(CPP) -c advdif2D.cpp $(OPT)

function2D.o: function2D.cpp function2D.h
	$(CPP) -c function2D.cpp $(OPT)

clean:
	rm *.o *.txt testAdvDif2D testDiffusion testFFT
