function LVGP_plot(model, varargin)

%% Parse the inputs
InParse = inputParser;
InParse.CaseSensitive = 0;
InParse.KeepUnmatched = 0;
InParse.PartialMatching = 1;
InParse.StructExpand = 1;

addRequired(InParse,'model');
addOptional(InParse, 'ind_qual_plot', []);

parse(InParse, model, varargin{:});

ind_qual_plot =  InParse.Results.ind_qual_plot;

ind_qual_all = model.data.ind_qual;

%% check indices of the variables to be plotted
if isempty(ind_qual_all)
    error('No qualitative/categorical variables in the model. No plot.');
end
if isempty(ind_qual_plot)
    ind_qual_plot = ind_qual_all;
    n_plot = length(ind_qual_plot);
    qual_only_ind = 1:n_plot;
else
    n_init = length(ind_qual_plot);
    ind_qual_plot = unique(ind_qual_plot);
    n_uniq = length(ind_qual_plot);
    if n_init ~= n_uniq
        warning('Ignoring duplicate indices specified.');
    end
    checkers = ismember(ind_qual_plot, ind_qual_all);
    if ~all(checkers)
        warning('Ignoring invalid indices.');
    end
    n_plot = sum(checkers);
    if n_plot == 0
        error('All indices input are invalid! No plot.');
    end
    ind_qual_plot = ind_qual_plot(checkers);
    temp = ismember(ind_qual_all, ind_qual_plot);
    qual_only_ind = find(temp); % the plot order might be reordered
end

%% load more data
dim_z = model.qual_param.dim_z;
z = model.qual_param.z;
lvs_qual = model.data.lvs_qual;
n_lvs_qual = model.data.n_lvs_qual;

%% plot
for i = 1:n_plot
    qual_only_ind_i = qual_only_ind(i);
    z_i = z{qual_only_ind_i};
    lvs_qual_i = round(lvs_qual{qual_only_ind_i},3,'significant');
    if dim_z == 1
        figure;
        plot(z_i, 0*ones(n_lvs_qual(qual_only_ind_i),1), 'o','markersize',5);
        text(z_i, -0.1*ones(n_lvs_qual(qual_only_ind_i),1), ...
            strtrim(cellstr(num2str(lvs_qual_i))));
        title(strcat('Latent variables for var #', num2str(ind_qual_plot(i))));
        xlabel('Latent variable value');
    elseif dim_z == 2
        x_range = max(z_i(:,1))-min(z_i(:,1));
        y_range = max(z_i(:,2))-min(z_i(:,2));
        figure;
        plot(z_i(:,1),z_i(:,2), 'o','markersize',5);
        text(z_i(:,1)-x_range*0.02, z_i(:,2)-y_range*0.02,...
            strtrim(cellstr(num2str(lvs_qual_i))));
        title(strcat('Latent variables for var #', num2str(ind_qual_plot(i))));
        xlabel('Latent variable 1 value');
        ylabel('Latent variable 2 value');
    else
        error('This plot function only works for 1D or 2D latent spaces!');
    end
end




















