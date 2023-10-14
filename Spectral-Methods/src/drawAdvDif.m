A = load("result.txt")';
N = size(A,2);
l = 0.5/N:1/N:1;
[x,y] = meshgrid(l, l);

pcolor(x, y, A);
shading flat
colorbar