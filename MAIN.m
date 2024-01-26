% Latent Variable Gaussian Process - Multi-Objective Bayesian Optimization 
% (LVGP-MOBBO) for combinatorial MOF Design Space Exploration

clearvars; close all;
rng('shuffle');
%% Add Required folders to the path
addpath('./LVGP/');
addpath('./MOBBO/');
addpath('./LVGP/utils/');
%% Read Design Candidates
datasource = 'fof_topology';
filename = strcat('./data/', datasource, '.csv');
data = table2array(readtable(filename));
X = data(:,3:6); % Candidate Designs with their vector represetations.

% Comment: Since only 1 topology is considered (fof topology), the 5th
% dimension (topology) can be excluded for the design representation. If 
% multiple topologies exits, 5th dimension can be added.
%% Objective Evaluator
obj = @(x) property_simulation_MOF(filename,x); % True (expensive) objective
% value calculator. Can be replaced with any expensive objective calculator
objs = struct('obj',{obj});

%% Defining LVGP Fitting options
fitoptions.ind_qual = [1,2,3,4]; % Columns of qualitative variables
fitoptions.dim_z = 2; % Dimension of latent variables
fitoptions.parallel = 1; % Parallel computing option: 1 for yes, 0 for no
seed = randi(1000,1); % assign random seed

%% Upload DOE and Create Candidate Design Search Space

% Load Initial Design of Experiments (DoE) designs
doe_data = load('./data/168_lhs_wc_selectivity_fof_1.mat'); % DoE data file location goes here
x0 = doe_data.X_int; % Initial Designs
y0 = doe_data.Y_int; % Initial Design Properties

% Batch Size for each MOBBO iteration  
k = 5;

% Remove DoE points from search space
x_can = X; % x_can is the candidate design space to search in
to_delete = ismember(X,x0,'rows'); % Identify initial DOE designs to remove
x_can(to_delete,:)=[]; % Remove initial DOE designs from the candidate space

%% Define Stopping Criteria
delta = 10^-5;
%% Run MOBBO
[x_PF_out, y_PF_out, x_eval, y_eval, ACF_val, UQmodels, y_pred, mse_pred] = ...
                      MOBO_MOF(x0, y0, x_can, seed, fitoptions, objs,k,delta);

%% Save Final LVGP-MOBBO Results
output_name = strcat('LVGP_MOBBO_Results'); % Output Filename
save(output_name,'x0','y0', 'x_PF_out', 'y_PF_out', 'x_eval', 'y_eval','ACF_val','UQmodels', 'y_pred', 'mse_pred');

