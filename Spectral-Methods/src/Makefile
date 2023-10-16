LIB=-lfftw3
OPT=-O2 -O3 -Ofast -msse -msse2 -msse3 -mmmx -m3dnow

all: testFFT testDiffusion testAdvDif2D

testAdvDif2D: fft2D.o diffusion2D.o matrix.o function2D.o advdif2D.o testAdvDif2D.o sparseMatrix.o
	g++ fft2D.o diffusion2D.o matrix.o function2D.o advdif2D.o testAdvDif2D.o -o testAdvDif2D sparseMatrix.o $(OPT) $(LIB)

testDiffusion: fft2D.o diffusion2D.o testDiffusion.o matrix.o function2D.o
	g++ fft2D.o diffusion2D.o testDiffusion.o matrix.o function2D.o -o testDiffusion $(OPT) $(LIB)

testFFT: fft2D.o testFFT.o
	g++ testFFT.o fft2D.o -o testFFT $(OPT) $(LIB)

testAdvDif2D.o: testAdvDif2D.cpp
	g++ -c testAdvDif2D.cpp $(OPT)

testDiffusion.o: testDiffusion.cpp
	g++ -c testDiffusion.cpp $(OPT)

testFFT.o: testFFT.cpp
	g++ -c testFFT.cpp $(OPT)

fft2D.o: fft2D.h fft2D.cpp
	g++ -c fft2D.cpp $(OPT)

diffusion2D.o: diffusion2D.h diffusion2D.cpp fft2D.h matrix.h function2D.h
	g++ -c diffusion2D.cpp $(OPT)

advdif2D.o: advdif2D.cpp advdif2D.h diffusion2D.h fft2D.h matrix.h function2D.h sparseMatrix.h
	g++ -c advdif2D.cpp $(OPT)

function2D.o: function2D.cpp function2D.h
	g++ -c function2D.cpp $(OPT)

matrix.o: matrix.h matrix.cpp
	g++ -c matrix.cpp $(OPT)

sparseMatrix.o: sparseMatrix.h sparseMatrix.cpp
	g++ -c sparseMatrix.cpp $(OPT)

clean:
	rm *.o *.txt testAdvDif2D testDiffusion testFFT