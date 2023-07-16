function [x_PF_out, y_PF_out, x_eval, y_eval, ACF_val,UQmodels, y_pred, mse_pred]  = MOBO_MOF(x0, y0, x_can, seed, fitoptions, objs,k,delta)

%{
Input variables:
x0 = n0-by-p matrix, inputs of the initial sample points
y0 = n0-by-q vector, outputs of the initial sample points
x_can = n-by-p matrix, inputs of the candidate sample points
n_steps = int, number of steps (new samples) in the search process
seed = int, seed of the random number generator
fitoptions = struct, options for fitting the LVGP model
objs = struct, (expensive) objective functions 
k = int,number of samples to add after each iteration

Output variables:
x_PF_out = ?-by-p matrix, inputs of the final PF points
y_PF_out = ?-by-q matrix, inputs of the final PF points
x_eval = designs evaluated during MOBBO
y_eval = objective values for x_eval (MOF Designs)
ACF_val = acquisition function value at each iteration
UQmodels = LVGP models saved during last iteration
y_pred = predicted value of objectives at each iteration
mse_pred = predicted variance of objectives at each iteration
%}

%% Initialization
close all;
rng(seed);
[n_all,~] = size(x_can);
q = size(y0, 2);

ACF_val = [];
y_pred = [];
mse_pred = [];

%% Fit initial LVGP models 
UQmodels = fitUQmodels(x0, y0, @LVGP_fit, fitoptions); %Initial LVGP fitting
%% Obtain initial Pareto Front
[y0_PF, x0_PF] = PFset(y0, x0); % Initial Pareto Front identification

%% MOBBO Initilization
inds_can = 1:n_all; % Total number of candidate designs
x_PF = x0_PF; % Paret front objective tracker
y_PF = y0_PF; % Pareto front design tracker
x_data = x0; % Explored design candidate tracker 
y_data = y0; % Explored design candidate objective tracker 
iter = 1; % Iteration counter

%% LVGP Predicction options
% This options enables LVGP to provide uncertainity prediction
predoptions.MSE_on = 1;

%% MOBBO Stopping Criteria
meanEMI = 1; % Initialize stopping criteria checker

%% MOBBO loop
while meanEMI>= delta
    n_can = length(inds_can);
    EMI_can = zeros(n_can,1);
    meanpred = zeros(n_can,2);
    msepred = zeros(n_can,2);

    % For all candidate designs, calculate the EMI values along with their
    % objective and variance predictions
    parfor j = 1:n_can
        [EMI_can(j), meanpred(j,:), msepred(j,:)] = EMI(UQmodels, ...
            @LVGP_predict, predoptions, x_can(inds_can(j),:), y_PF);
    end
    % Identify the best batch of designs with highest EMI values
    [EMI_best,id_best] = maxk(EMI_can,k); 
    ind_best = inds_can(id_best);
    x_best = x_can(ind_best,:);

    % Stroring LVGP predictions of best designs
    y_pred =  [y_pred;meanpred(id_best,:)];
    mse_pred = [mse_pred;msepred(id_best,:)];
    
    % Calculate the mean of Expected Improvement values for stopping
    % criteria
    meanEMI = mean(EMI_best);

    % Obtain the objective values at the suggested design
    y_best = evalobjs(x_best, objs);
    
    % Update the LVGP models by re-training 
    x_data = [x_data; x_best]; % Update the inputs 
    y_data = [y_data; y_best]; % Update the outputs
    UQmodels = fitUQmodels(x_data, y_data, @LVGP_fit, fitoptions); 

    % Pareto Front is updated with the new designs
    for p = 1:size(x_best,1)
        [y_PF, x_PF] = updatePF(y_best(p,:),x_best(p,:), y_PF, x_PF);
    end
    
    % Update the candidate designs by removing selected designs
    inds_can(id_best) = [];
    
    % Printing information regarding the MOBBO
    fprintf('Completed Iteration # %d.', iter)
    save(['LVGP_MOBBO_iteration_',num2str(iter),'.mat']) %option to save information in each MOBBO iteration
    iter = iter + 1;
end


%% Outputing Final Pareto Front and Explored Candidates
ntrain = size(x0,1);
ntot = size(x_data,1);
x_eval = x_data(ntrain+1:ntot,:); % Final explored candidates
y_eval = y_data(ntrain+1:ntot,:); % Final explored candidate objective values
x_PF_out = x_PF; % Final Pareto front designs
y_PF_out = y_PF; % Final Pareto front objective values

end


