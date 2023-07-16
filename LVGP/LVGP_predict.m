function pred = LVGP_predict(X_new, model, varargin)

%% Parse the inputs
InParse = inputParser;
InParse.CaseSensitive = 0;
InParse.KeepUnmatched = 0;
InParse.PartialMatching = 1;
InParse.StructExpand = 1;

vf1 = @(x) isnumeric(x) && isreal(x);
addRequired(InParse,'X_new', vf1);
addRequired(InParse,'model');
addOptional(InParse, 'MSE_on', false);

parse(InParse, X_new, model, varargin{:});

MSE_on =  InParse.Results.MSE_on;

%% load model data
p_all = model.data.p_all;
p_qual = model.data.p_qual;
%p_quant = model.data.p_quant;

X_quant_min = model.data.X_quant_min;
X_quant_max = model.data.X_quant_max;
Y_min = model.data.Y_min;
Y_max = model.data.Y_max;

X_old_full = model.data.X_full;
lvs_qual = model.data.lvs_qual;
n_lvs_qual = model.data.n_lvs_qual;
ind_qual = model.data.ind_qual;

phi = model.quant_param.phi;
dim_z = model.qual_param.dim_z;
z_vec = model.qual_param.z_vec;

beta_hat = model.fit_detail.beta_hat;
RinvPYminusMbetaP = model.fit_detail.RinvPYminusMbetaP;

%% Process X_new
[m, pp] = size(X_new);
assert(p_all==pp,'The dimensionality of X_new is incorrect');

if p_qual == 0
    %X_new_qual = [];
    X_new_quant = X_new;
    X_new_quant = (X_new_quant-repmat(X_quant_min, m, 1))./...
        repmat(X_quant_max-X_quant_min, m, 1);
    R_old_new = corr_mat(X_old_full, X_new_quant, phi);
else
    X_new_qual = X_new(:,ind_qual);
    if p_qual == p_all
        X_new_quant = [];
    else
        X_new_quant = X_new(:,:);
        X_new_quant(:,ind_qual) = [];
        X_new_quant = (X_new_quant-repmat(X_quant_min, m, 1))./...
            repmat(X_quant_max-X_quant_min, m, 1);
    end
    X_new_qual_la = to_latent(X_new_qual, lvs_qual, n_lvs_qual, p_qual, ...
        z_vec, dim_z, m);
    X_new_full = [X_new_quant, X_new_qual_la];
    phi_full = [phi, zeros(1, p_qual*dim_z)];
    R_old_new = corr_mat(X_old_full, X_new_full, phi_full);
end


%% Calc predictions
Y_hat = beta_hat + R_old_new'*RinvPYminusMbetaP;
Y_hat = Y_hat*(Y_max-Y_min)+Y_min;

pred.Y_hat = Y_hat;

if MSE_on
    if p_qual == 0
        R_new_new = corr_mat(X_new_quant, X_new_quant, phi);
    else
        R_new_new = corr_mat(X_new_full, X_new_full, phi_full);
    end
    R_new_new = 0.5*(R_new_new+R_new_new');
    
    sigma2 = model.fit_detail.sigma2;
    Linv = model.fit_detail.Linv;
    MTRinvM = model.fit_detail.MTRinvM;
    LinvM = model.fit_detail.LinvM;
    
    temp = Linv*R_old_new;
    W = 1- LinvM'*temp;
    MSE = sigma2*(R_new_new-temp'*temp+W'*W/MTRinvM)*((Y_max-Y_min)^2);
    pred.MSE = MSE;
end

end




















