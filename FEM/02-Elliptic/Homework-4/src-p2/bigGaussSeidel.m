Nj = [128, 512, 2048, 8192, 32768];

id_i = zeros(1, 10000000);
id_j = zeros(1, 10000000);
vals = zeros(1, 10000000);
nonzeros = 0;

for j = 1 : 5
    N = Nj(j);
    for k = 0 : N
        addElement(idx(j,k), idx(j,k), 2*N);
        addElement(idx(j,k), idx(j,k+1), -N);
        addElement(idx(j,k), idx(j,k-1), -N);
        p = 1;
        for fj = (j+1) : 5
            p = p * 4;
            addElement(idx(j,k), idx(fj,k*p), 2*N);
            addElement(idx(fj,k*p), idx(j,k), 2*N);
        end
    end
end

function [] = addElement(j, k, val)
    nonzeros = nonzeros + 1;
    id_i(nonzeros) = j;
    id_j(nonzeros) = k;
    vals(nonzeros) = val;
end