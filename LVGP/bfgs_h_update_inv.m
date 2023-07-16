function out = bfgs_h_update_inv(H_k, y_k, s_k)

Hkyk = H_k * y_k;
skyk = s_k'*y_k;
ykHkyk = y_k'*Hkyk;

if abs(ykHkyk) > 1e-8 && abs(skyk) > 1e-8 % update hessian
    
    theta_k = 1;    % default value for theta_k is 1
    if skyk < 0.2*ykHkyk
        theta_k = (0.8*ykHkyk)/(ykHkyk-skyk);
    end

    r_k = theta_k*s_k + (1-theta_k)*Hkyk;
    rho_k = 1/(r_k'*y_k);
    rkHkyk = r_k*Hkyk';
    out = H_k - rho_k*(rkHkyk + rkHkyk' - (rho_k * ykHkyk + 1)*(r_k*r_k'));

else
    out = H_k;
    %disp('No need to update hessian');
end

end