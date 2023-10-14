function merr = maxerr(corset, fine)
    n = size(corset,1);
    merr = 0;
    for i = 1:n
        for j = 1:n
            v = 0.25 * (fine(2*i-1,2*j-1) + fine(2*i-1,2*j) + fine(2*i,2*j-1) + fine(2*i,2*j));
            merr = max([merr, abs(corset(i,j) - v)]);
        end
    end
end