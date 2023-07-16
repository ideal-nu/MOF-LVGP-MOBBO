function model = LVGP_fit(X, Y, varargin)

%% Parse the inputs
InParse = inputParser;
InParse.CaseSensitive = 0;
InParse.KeepUnmatched = 0;
InParse.PartialMatching = 1;
InParse.StructExpand = 1;

vf1 = @(x) isnumeric(x) && isreal(x);
addRequired(InParse,'X', vf1);
addRequired(InParse,'Y', vf1);
addOptional(InParse, 'ind_qual', []);
addOptional(InParse, 'dim_z', 2);
addOptional(InParse, 'eps_all', 10.^(-1:-0.5:-8)');
addOptional(InParse, 'lb_phi_ini', -2);
addOptional(InParse, 'ub_phi_ini', 2);
addOptional(InParse, 'lb_phi_lat', -8);
addOptional(InParse, 'ub_phi_lat', 8);
addOptional(InParse, 'lb_z', -3); %-3
addOptional(InParse, 'ub_z', 3); % 3
addOptional(InParse, 'n_opt', 8);
addOptional(InParse, 'parallel', 0);
addOptional(InParse, 'max_iter_ini',100); %100
addOptional(InParse, 'max_iter_lat', 20); %20
addOptional(InParse, 'max_eval_ini', 5000); %5000
addOptional(InParse, 'max_eval_lat', 1000); %1000
addOptional(InParse, 'seed', 1505);
addOptional(InParse, 'progress', false);
addOptional(InParse, 'noise', false);
addOptional(InParse, 'tol', 10^-6);

parse(InParse, X, Y, varargin{:});

ind_qual =  InParse.Results.ind_qual;
dim_z =  InParse.Results.dim_z;
eps_all =  InParse.Results.eps_all;
lb_phi_ini =  InParse.Results.lb_phi_ini;
ub_phi_ini =  InParse.Results.ub_phi_ini;
lb_phi_lat =  InParse.Results.lb_phi_lat;
ub_phi_lat =  InParse.Results.ub_phi_lat;
lb_z =  InParse.Results.lb_z;
ub_z =  InParse.Results.ub_z;
n_opt =  InParse.Results.n_opt;
parallel =  InParse.Results.parallel;
max_iter_ini =  InParse.Results.max_iter_ini;
max_iter_lat =  InParse.Results.max_iter_lat;
max_eval_ini =  InParse.Results.max_eval_ini;
max_eval_lat =  InParse.Results.max_eval_lat;
seed =  InParse.Results.seed;
progress =  InParse.Results.progress;
noise =  InParse.Results.noise;
tol = InParse.Results.tol;


%% Global initialization and variable check
rng(seed);
if progress
    disp('** Checking & preprocessing the inputs...');
end

[k, p_all] = size(X);

if isempty(ind_qual)
    p_qual = 0;
    X_qual = [];
    X_quant = X;
    lvs_qual = [];
    n_lvs_qual = [];
    n_z = 0;
else
    p_qual = length(ind_qual);
    X_qual = X(:,ind_qual);
    if p_qual == p_all
        X_quant = [];
    else
        X_quant = X(:,:);
        X_quant(:,ind_qual) = [];
    end
    
    lvs_qual = cell(p_qual, 1);
    n_lvs_qual = zeros(1, p_qual);
    for i = 1:p_qual
        lvs_qual{i} = sort(unique(X_qual(:,i)));
        n_lvs_qual(i) = length(lvs_qual{i});
    end
    n_z = dim_z*(sum(n_lvs_qual)-p_qual);
end

assert(k==size(Y,1),'The number of rows of X and Y must match');
if dim_z ~= 1 && dim_z ~= 2
    warning('The dimensionality of latent space is uncommon!');
end
eps_min_allow = 10^(round(log10(eps)+3));
if any(eps_all < (eps_min_allow))
    error(strcat('The minimum allowed eps is ',num2str(eps_min_allow)));
end
if any(diff(eps_all)>0)
    error('The elements in eps_all should be in descending order.');
end

p_quant = p_all - p_qual;

%% Normalization of X and Y
if p_quant > 0
    X_quant_min = min(X_quant); X_quant_max = max(X_quant);
    X_quant = (X_quant-repmat(X_quant_min, k, 1))./...
        repmat(X_quant_max-X_quant_min, k, 1);
else
    X_quant_min = [];
    X_quant_max = [];
end
Y_min = min(Y); Y_max = max(Y);
Y = (Y-Y_min)/(Y_max-Y_min);

%% Initialization for optimization
if progress
    disp('** Initializing optimization procedure...');
end
n_hyper = p_quant + n_z;
lb_ini = [lb_phi_ini*ones(1, p_quant), lb_z*ones(1,n_z)];
ub_ini = [ub_phi_ini*ones(1, p_quant), ub_z*ones(1,n_z)];
lb_lat = [lb_phi_lat*ones(1, p_quant), lb_z*ones(1,n_z)];
ub_lat = [ub_phi_lat*ones(1, p_quant), ub_z*ones(1,n_z)];
if dim_z == 2 && p_qual ~= 0
    temp_ind = p_quant;
    for i = 1:p_qual
        n_lvs = n_lvs_qual(i);
        lb_ini(temp_ind+dim_z) = -1e-4;
        ub_ini(temp_ind+dim_z) = 1e-4;
        lb_lat(temp_ind+dim_z) = -1e-4;
        ub_lat(temp_ind+dim_z) = 1e-4;
        temp_ind = temp_ind + dim_z*(n_lvs-1);
    end
end

options_ini = optimoptions(@fmincon, 'Algorithm', 'interior-point', ...
    'display', 'off', 'MaxIterations', max_iter_ini,...
    'MaxFunctionEvaluations', max_eval_ini,'OptimalityTolerance',tol);
    %'ScaleProblem','obj-and-constr',...
    %'FiniteDifferenceType', 'central');
options_lat = optimoptions(@fmincon, 'Algorithm', 'interior-point', ...
    'display', 'off', 'MaxIterations', max_iter_lat,...
    'MaxFunctionEvaluations', max_eval_lat,'OptimalityTolerance',tol);
    %'ScaleProblem','obj-and-constr',...
    %'FiniteDifferenceType', 'central');

setting = struct('max_iter_ini', max_iter_ini, 'max_iter_lat',max_iter_lat,...
    'max_func_eval_ini', max_eval_ini,...
    'max_func_eval_lat', max_eval_lat,...
    'seed', seed, 'n_opt', n_opt, 'lb_phi_ini', lb_phi_ini, ...
    'ub_phi_ini', ub_phi_ini, 'lb_phi_lat', lb_phi_lat,...
    'ub_phi_lat', ub_phi_lat, 'lb_z', lb_z, 'ub_z',ub_z, 'progress',progress,...
    'parallel',parallel, 'eps_all',eps_all,'tol',tol);

%% optimization
tic;
if progress
    disp('** Optimization started ...');
end
A = sobolset(n_hyper,'Skip',1e3);
A = scramble(A,'MatousekAffineOwen');
hyper0 = A(1:n_opt, :).*repmat(ub_ini-lb_ini, n_opt, 1)+...
    repmat(lb_ini, n_opt, 1);
M = ones(k, 1);

n_try = length(eps_all);
optim_hist = struct('hyper0',cell(n_try,1), 'hyper_sol', cell(n_try,1),...
    'hyper_sol_uni', cell(n_try,1), 'obj_sol', cell(n_try,1),...
    'obj_sol_uni', cell(n_try,1), 'obj_sol_best',cell(n_try,1),...
    'flag_sol_uni',cell(n_try,1));
obj_sol_best = zeros(n_try, 1);

gradient = zeros(n_opt, n_hyper);
hessian_ini = zeros(n_hyper, n_hyper, n_opt); 
hessian_inv = zeros(n_hyper, n_hyper, n_opt); 
d_const = 1e-8;

for i_try = 1:n_try
    eps_i = eps_all(i_try);
    n_opt_i = size(hyper0, 1);
    hyper_sol = zeros(n_opt_i, n_hyper);
    obj_sol = zeros(n_opt_i, 1);
    flag_sol = zeros(n_opt_i, 1);
    if i_try == 1
        if parallel && (size(hyper0, 1) > 1)
            parfor i = 1:n_opt_i
                [hyper_sol(i,:), obj_sol(i,1), flag_sol(i,1),...
                    ~, ~, gradient(i, :), hessian_ini(:, :, i)] = fmincon(...
                    @(x) neg_log_l(x, p_quant, p_qual, lvs_qual, n_lvs_qual,...
                    dim_z, X_quant, X_qual, Y, eps_i, k, M, n_z), hyper0(i,:),...
                    [], [], [], [], lb_ini, ub_ini, [], options_ini);
                hessian_inv(:, :, i) = inv(hessian_ini(:, :, i));
            end
        else
            for i = 1:n_opt_i
                [hyper_sol(i,:), obj_sol(i,1), flag_sol(i,1),...
                    ~, ~, gradient(i, :), hessian_ini(:, :, i)] = fmincon(...
                    @(x) neg_log_l(x, p_quant, p_qual, lvs_qual, n_lvs_qual,...
                    dim_z, X_quant, X_qual, Y, eps_i, k, M, n_z), hyper0(i,:),...
                    [], [], [], [], lb_ini, ub_ini, [], options_ini);
                hessian_inv(:, :, i) = inv(hessian_ini(:, :, i));
            end
        end
        
    else % BFGS UPDATE
        %d_delta = eps_all(i_try) - eps_all(i_try - 1);
        hyper_sol = zeros(n_opt_i, n_hyper);
        ub_b_vec_all = repmat(ub_lat, n_opt_i, 1) - hyper0;
        lb_b_vec_all = repmat(lb_lat, n_opt_i, 1) - hyper0;
        f = @(params) neg_log_l(params, p_quant, p_qual, lvs_qual, n_lvs_qual,...
            dim_z, X_quant, X_qual, Y, eps_i, k, M, n_z);
        if parallel && (size(hyper0, 1) > 1)
            parfor i = 1:n_opt_i
                inter_grad = (my_grad(f, hyper0(i,:), d_const))';
                
                %pre_y_k = inter_grad - comp_gradient(i, :)';
                %pre_s_k = [zeros(n_hyper, 1); d_delta];
                %inter_hess = bfgs_h_update(comp_hessian(:, :, i), pre_y_k, pre_s_k);
                
                search_dir = -hessian_inv(:, :, i) * inter_grad;
                %search_dir = search_dir(1:end - 1);                
                min_ub = inf;
                max_lb = -inf;
                for d = 1:n_hyper
                    if search_dir(d) > 0
                        if abs(ub_b_vec_all(i,d)/search_dir(d)) < 1e-4
                            search_dir(d) = 0;
                        else
                            min_ub = min(min_ub, ub_b_vec_all(i,d)/search_dir(d));
                        end
                        max_lb = max(max_lb, lb_b_vec_all(i,d)/search_dir(d));
                    elseif search_dir(d) < 0
                        max_lb = max(max_lb, ub_b_vec_all(i,d)/search_dir(d));
                        if abs(lb_b_vec_all(i,d)/search_dir(d)) < 1e-4
                            search_dir(d) = 0;
                        else
                            min_ub = min(min_ub, lb_b_vec_all(i,d)/search_dir(d));
                        end
                    end
                end
                %if max_lb > 0
                %    error('This should not happen');
                %end
                tmp_obj_func = @(alpha) f(hyper0(i,:) + search_dir'*alpha);
                alpha_opt = linesearch(tmp_obj_func, ...
                    inter_grad, search_dir, min_ub);
                
                s_k = search_dir*alpha_opt;
                hyper_sol(i,:) = hyper0(i,:) + s_k';
                new_grad = my_grad(f, hyper_sol(i,:), d_const);
                y_k = new_grad' - gradient(i, :)';
                
                % update grad and hessian
                gradient(i, :) = new_grad;
                hessian_inv(:, :, i) = bfgs_h_update_inv(hessian_inv(:, :, i), y_k, s_k);
            end
        else
            for i = 1:n_opt_i
                inter_grad = (my_grad(f, hyper0(i,:), d_const))';
                
                %pre_y_k = inter_grad - comp_gradient(i, :)';
                %pre_s_k = [zeros(n_hyper, 1); d_delta];
                %inter_hess = bfgs_h_update(comp_hessian(:, :, i), pre_y_k, pre_s_k);
                
                search_dir = -hessian_inv(:, :, i) * inter_grad;
                %search_dir = search_dir(1:end - 1);                
                min_ub = inf;
                max_lb = -inf;
                for d = 1:n_hyper
                    if search_dir(d) > 0
                        if abs(ub_b_vec_all(i,d)/search_dir(d)) < 1e-4
                            search_dir(d) = 0;
                        else
                            min_ub = min(min_ub, ub_b_vec_all(i,d)/search_dir(d));
                        end
                        max_lb = max(max_lb, lb_b_vec_all(i,d)/search_dir(d));
                    elseif search_dir(d) < 0
                        max_lb = max(max_lb, ub_b_vec_all(i,d)/search_dir(d));
                        if abs(lb_b_vec_all(i,d)/search_dir(d)) < 1e-4
                            search_dir(d) = 0;
                        else
                            min_ub = min(min_ub, lb_b_vec_all(i,d)/search_dir(d));
                        end
                    end
                end
                %if max_lb > 0
                %    error('This should not happen');
                %end
                tmp_obj_func = @(alpha) f(hyper0(i,:) + search_dir'*alpha);
                alpha_opt = linesearch(tmp_obj_func, ...
                    inter_grad, search_dir, min_ub);
                
                s_k = search_dir*alpha_opt;
                hyper_sol(i,:) = hyper0(i,:) + s_k';
                new_grad = my_grad(f, hyper_sol(i,:), d_const);
                y_k = new_grad' - gradient(i, :)';
                
                % update grad and hessian
                gradient(i, :) = new_grad;
                hessian_inv(:, :, i) = bfgs_h_update_inv(hessian_inv(:, :, i), y_k, s_k);
            end
        end
        optim_hist(i_try).hyper0 = hyper0;
        optim_hist(i_try).hyper_sol = hyper_sol;
        hyper0 = hyper_sol;
    end
    if i_try == n_try % final refinement
        if parallel && (size(hyper0, 1) > 1)
            parfor i = 1:n_opt_i
                [hyper_sol(i,:), obj_sol(i,1), flag_sol(i,1)] = fmincon(...
                    @(x) neg_log_l(x, p_quant, p_qual, lvs_qual, n_lvs_qual,...
                    dim_z, X_quant, X_qual, Y, eps_i, k, M, n_z), hyper0(i,:),...
                    [], [], [], [], lb_lat, ub_lat, [], options_lat);
            end
        else
            for i = 1:n_opt_i
                [hyper_sol(i,:), obj_sol(i,1), flag_sol(i,1)] = fmincon(...
                    @(x) neg_log_l(x, p_quant, p_qual, lvs_qual, n_lvs_qual,...
                    dim_z, X_quant, X_qual, Y, eps_i, k, M, n_z), hyper0(i,:),...
                    [], [], [], [], lb_lat, ub_lat, [], options_lat);
            end
        end
    end
    
    if i_try == 1 || i_try == n_try
        [obj_sol, ID] = sort(obj_sol);
        flag_sol = flag_sol(ID);
        hyper_sol = hyper_sol(ID,:);
        hyper0 = hyper0(ID,:);
        gradient = gradient(ID, :);
        hessian_inv = hessian_inv(:, :, ID);

        % find unique solutions
        [obj_sol_uni, uni_ID] = unique(round(obj_sol, 2));
        flag_sol_uni = flag_sol(uni_ID);
        hyper_sol_uni = hyper_sol(uni_ID, :);

        optim_hist(i_try).hyper0 = hyper0;
        optim_hist(i_try).hyper_sol = hyper_sol;
        optim_hist(i_try).hyper_sol_uni = hyper_sol_uni;
        optim_hist(i_try).obj_sol = obj_sol;
        optim_hist(i_try).obj_sol_uni = obj_sol_uni;
        optim_hist(i_try).obj_sol_best = obj_sol(1);
        obj_sol_best(i_try) = obj_sol(1);
        optim_hist(i_try).flag_sol_uni = flag_sol_uni;

        if i_try < n_try
            hyper0 = hyper_sol_uni;
            gradient = gradient(uni_ID, :);
            hessian_inv = hessian_inv(:, :, uni_ID);
        end
    end
end

fit_time = toc;

%% Post-processing
if progress
    disp('** Post-processing ...');
end
if ~noise
    id_best_try = n_try;
else
    [~, id_best_try] = min(obj_sol_best);
end

hyper_full = optim_hist(id_best_try).hyper_sol(1,:);
min_n_log_l = obj_sol_best(id_best_try);

if p_quant == 0
    phi = [];
else
    phi = hyper_full(1:p_quant);
end
if p_qual == 0
    z_vec = []; z = [];
else
    z_vec = hyper_full((p_quant+1):(p_quant+n_z));
    z = cell(p_qual, 1);
    ind_temp = 0;
    for i = 1:p_qual
        n_lvs = n_lvs_qual(i);
        z_i = z_vec((dim_z*ind_temp+1):(dim_z*(ind_temp+n_lvs-1)));
        ind_temp = ind_temp + n_lvs-1;
        z{i} = [zeros(1,dim_z);reshape(z_i, dim_z, n_lvs-1)'];
    end  
end

% calc convenient quantities
if p_qual == 0
    R = corr_mat(X_quant, X_quant, phi);
    X_full = X_quant;
else
    X_qual_la = to_latent(X_qual, lvs_qual, n_lvs_qual, p_qual, z_vec, dim_z, k);
    X_full = [X_quant, X_qual_la];
    phi_full = [phi, zeros(1, p_qual*dim_z)];
    R = corr_mat(X_full, X_full, phi_full);
end

R = 0.5*(R+R');

raw_min_eig = min(eig(R));
if raw_min_eig < eps_all(id_best_try)
    nug_opt = eps_all(id_best_try)-raw_min_eig;
    R = R + eye(k,k)*nug_opt;
else
    nug_opt = 0;
end

L = chol(R,'lower'); % L*L' = R
Linv = inv(L);
LinvM = L\M;
MTRinvM = sum(LinvM.^2);
beta_hat = (LinvM'*(L\Y))/MTRinvM;
temp = L\(Y-M*beta_hat);
sigma2 = sum(temp.^2)/k;
if sigma2 < 1e-300
    sigma2 = 1e-300;
end
RinvPYminusMbetaP = L'\temp;

%% model selection criteria
temp2 = RinvPYminusMbetaP./diag(Linv'*Linv);
LOOCV_L1 = mean(abs(temp2));
LOOCV_L2 = mean(temp2.^2);
if dim_z == 1
    AIC = (min_n_log_l+1/k+k*log(2*pi))/2 + 2*n_hyper;
    BIC = (min_n_log_l+1/k+k*log(2*pi))/2 + log(k)*n_hyper;
elseif dim_z == 2 % dim_z == 2. One degree of freedom fixed
    AIC = (min_n_log_l+1/k+k*log(2*pi))/2 + 2*(n_hyper-p_qual);
    BIC = (min_n_log_l+1/k+k*log(2*pi))/2 + log(k)*(n_hyper-p_qual);
end

%% Save the fitted model
model = struct();
model.quant_param = struct('phi',phi, 'lb_phi_ini', lb_phi_ini,...
    'ub_phi_ini', ub_phi_ini, 'lb_phi_lat',lb_phi_lat,...
    'ub_phi_lat', ub_phi_lat);
model.qual_param = struct('dim_z',dim_z, 'z',{z}, 'z_vec',z_vec,...
    'lb_z',lb_z, 'ub_z',ub_z);
model.data = struct('X', X, 'X_quant',X_quant, 'X_qual',X_qual,...
    'X_full', X_full, 'Y', Y, 'X_quant_min', X_quant_min,...
    'X_quant_max', X_quant_max, 'Y_min', Y_min, 'Y_max', Y_max,...
    'ind_qual', ind_qual, 'lvs_qual', {lvs_qual}, 'n_lvs_qual',n_lvs_qual,...
    'p_all', p_all, 'p_quant', p_quant, 'p_qual', p_qual);
model.fit_detail = struct('beta_hat',beta_hat, 'sigma2', sigma2, ...
    'MTRinvM', MTRinvM, 'Linv', Linv, 'LinvM', LinvM, ...
    'RinvPYminusMbetaP', RinvPYminusMbetaP, 'raw_min_eig', raw_min_eig,...
    'nug_opt',nug_opt, 'min_n_log_l', min_n_log_l, 'fit_time', fit_time,...
    'LOOCV_L1', LOOCV_L1, 'LOOCV_L2', LOOCV_L2, 'AIC', AIC, 'BIC', BIC);
model.optim_hist = optim_hist;
model.setting = setting;

end




















