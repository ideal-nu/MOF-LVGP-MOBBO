function X_qual_la = to_latent(X_qual, lvs_qual, n_lvs_qual, p_qual, z_vec, dim_z, k)
%{
inputs:
    X_qual: Matrix containing (only) the qualitative/categorical data.
    lvs_qual: Cell vector containing levels of each qualitative variable
    n_lvs_qual: Number of levels of each qualitative variable
    p_qual: Number of qualitative variables
    z_vec: Latent variable parameters, i.e., latent variable values for ...
        each level of qualitative/categorical variables
    dim_z: Dimensionality of latent variables, usually 1 or 2
    k: Number of data points
%}
X_qual_la = zeros(k, p_qual*dim_z);
%note: the first levels of each variable are fixed as zeros ...
%   in the latent space.
ind_temp = 0;
for i = 1:p_qual
    n_lvs = n_lvs_qual(i);
    lvs = lvs_qual{i};
    z_i = z_vec(dim_z*ind_temp+1:(dim_z*(ind_temp+n_lvs-1)));
    ind_temp = ind_temp+n_lvs-1;
    for j = 2:n_lvs
        mask = (X_qual(:,i)==lvs(j));
        num_row = sum(mask);
        if num_row >0
            X_qual_la(mask, ((i-1)*dim_z+1):(i*dim_z))=...
                repmat(z_i((dim_z*(j-2)+1):(dim_z*(j-1))), num_row,1);
        end
    end
end

end


















