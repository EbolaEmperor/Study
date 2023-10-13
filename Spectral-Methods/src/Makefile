all: testFFT testDiffusion

testDiffusion: fft2D.o diffusion2D.o testDiffusion.o
	g++ fft2D.o diffusion2D.o testDiffusion.o -o testDiffusion -O2 -O3 -Ofast

testFFT: fft2D.o testFFT.o
	g++ testFFT.o fft2D.o -o testFFT -O2 -O3 -Ofast

testDiffusion.o: testDiffusion.cpp
	g++ -c testDiffusion.cpp -O2 -O3 -Ofast

testFFT.o: testFFT.cpp
	g++ -c testFFT.cpp -O2 -O3 -Ofast

fft2D.o: fft2D.h fft2D.cpp
	g++ -c fft2D.cpp -O2 -O3 -Ofast

diffusion2D.o: diffusion2D.h diffusion2D.cpp fft2D.h
	g++ -c diffusion2D.cpp -O2 -O3 -Ofast

clean:
	rm *.o