A = load("result-dense.txt");
N = sqrt(size(A,2));
l = 0.5/N:1/N:1;
[x,y] = meshgrid(l, l);

pcolor(x, y, reshape(A(4,:),N,N));
shading flat
colorbar