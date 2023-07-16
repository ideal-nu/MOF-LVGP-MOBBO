%% PF set function
function [y_PF, aux_PF] = PFset(y_all, aux_all)

[n_x, ~] = size(y_all);
y_PF = y_all(1,:);
aux_PF = aux_all(1,:);
for i = 2:n_x
    y_candid = y_all(i,:);
    aux_candid = aux_all(i,:);
    [y_PF, aux_PF] = updatePF(y_candid, aux_candid, y_PF, aux_PF);
end

end