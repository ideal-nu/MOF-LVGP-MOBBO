function grad = my_grad(func, x0, stepsize)

%stepsize = 1e-8;
x_len = length(x0);
grad = zeros(1, x_len);

y_orig = func(x0);

for i = 1:x_len
    x_new = x0;
    x_new(i) = x_new(i) + stepsize;
    y_new = func(x_new);
    grad(i) = (y_new - y_orig)/stepsize;
end

end