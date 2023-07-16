function alpha = linesearch(objfunc, gradient_k, p_k, ub)
rho = 0.5;
c1 = 1e-4;
alpha_l = min(1e-8, ub/10.0);

f_k = objfunc(0);

alpha = min(1, ub);
f_trial = objfunc(alpha);

while (alpha > alpha_l && f_trial > f_k + c1*alpha*(gradient_k'*p_k))
    alpha = rho*alpha;  % decrease alpha when sufficient decrease condition isn't met
    f_trial = objfunc(alpha);
end

end