all: testFFT testDiffusion testAdvDif2D

testAdvDif2D: fft2D.o diffusion2D.o matrix.o function2D.o advdif2D.o testAdvDif2D.o
	g++ fft2D.o diffusion2D.o matrix.o function2D.o advdif2D.o testAdvDif2D.o -o testAdvDif2D -O2 -O3 -Ofast

testDiffusion: fft2D.o diffusion2D.o testDiffusion.o matrix.o function2D.o
	g++ fft2D.o diffusion2D.o testDiffusion.o matrix.o function2D.o -o testDiffusion -O2 -O3 -Ofast

testFFT: fft2D.o testFFT.o
	g++ testFFT.o fft2D.o -o testFFT -O2 -O3 -Ofast

testAdvDif2D.o: testAdvDif2D.cpp
	g++ -c testAdvDif2D.cpp -O2 -O3 -Ofast

testDiffusion.o: testDiffusion.cpp
	g++ -c testDiffusion.cpp -O2 -O3 -Ofast

testFFT.o: testFFT.cpp
	g++ -c testFFT.cpp -O2 -O3 -Ofast

fft2D.o: fft2D.h fft2D.cpp
	g++ -c fft2D.cpp -O2 -O3 -Ofast

diffusion2D.o: diffusion2D.h diffusion2D.cpp fft2D.h matrix.h function2D.h
	g++ -c diffusion2D.cpp -O2 -O3 -Ofast

advdif2D.o: advdif2D.cpp advdif2D.h diffusion2D.h fft2D.h matrix.h function2D.h
	g++ -c advdif2D.cpp -O2 -O3 -Ofast

function2D.o: function2D.cpp function2D.h
	g++ -c function2D.cpp -O2 -O3 -Ofast

matrix.o: matrix.h matrix.cpp
	g++ -c matrix.cpp -O2 -O3 -Ofast

clean:
	rm *.o *.txt testAdvDif2D testDiffusion testFFT