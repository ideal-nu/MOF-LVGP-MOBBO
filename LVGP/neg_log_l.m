function output = neg_log_l(hyperparam, p_quant, p_qual, ...
lvs_qual, n_lvs_qual, dim_z, X_quant, X_qual, Y, min_eig, k, M, n_z)
%{
inputs:
    hyperparam: Hyperparameters of the LVGP model
    p_quant: Number of quantative variables
    p_qual: Number of qualitative variables
    lvs_qual: Levels of each qualitative variable
    n_lvs_qual: Number of levels of each qualitative variable
    dim_z: Dimensionality of latent variables, usually 1 or 2
    X_quant: Input data of quantative variables
    X_qual: Input data of qualitative variables
    Y Vector: containing the outputs of data points
    min_eig: The smallest eigen value that the correlation matrix is allowed ...
        to have, which determines the nugget added to the correlation matrix.
    k: Number of data points
    M: Vector of ones with length k
    n_z: Number of latent parameters
%}

if p_qual == 0
    R = corr_mat(X_quant, X_quant, hyperparam);
else
    z_vec = hyperparam((p_quant+1):(p_quant+n_z));
    X_qual_la = to_latent(X_qual, lvs_qual, n_lvs_qual, p_qual, z_vec, dim_z, k);
    X_full = [X_quant, X_qual_la];
    phi_full = [hyperparam(1:p_quant), zeros(1, p_qual*dim_z)];
    R = corr_mat(X_full, X_full, phi_full);
end
R = 0.5*(R+R');

raw_min_eig = min(eig(R));
if raw_min_eig < min_eig
    R = R + eye(k,k)*(min_eig-raw_min_eig);
end

L = chol(R,'lower'); % L*L' = R
LinvM = L\M;
beta_hat = (LinvM'*(L\Y))/sum(LinvM.^2);
temp = L\(Y-M*beta_hat);
sigma2 = sum(temp.^2)/k;
if sigma2 < 1e-300
    sigma2 = 1e-300;
end

output = k*log(sigma2)+2*sum(log(diag(L)));

end


















