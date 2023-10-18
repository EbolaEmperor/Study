function merr = err(corset, fine, k)
    n = size(corset,1);
    merr = 0;
    for i = 1:n
        for j = 1:n
            v = 0.25 * (fine(2*i-1,2*j-1) + fine(2*i-1,2*j) + fine(2*i,2*j-1) + fine(2*i,2*j));
            merr = merr + abs(corset(i,j) - v)^k;
        end
    end
    merr = (merr/(n*n))^(1/k);
end