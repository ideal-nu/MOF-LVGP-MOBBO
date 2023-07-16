%% update PF set with new point
function [y_PF, aux_PF] = updatePF(y_new, aux_new, y_PF, aux_PF)

n_PF = size(y_PF, 1);
diff = y_PF - repmat(y_new, n_PF, 1);
% check if any point in x_PF is dominated by new point. Remove them.
checker_old = all(diff'>0);
if any(checker_old)
    y_PF(checker_old,:) = [];
    aux_PF(checker_old,:) = [];
    % add the new point to PF set
    y_PF = [y_PF; y_new];
    aux_PF = [aux_PF; aux_new];
else %check if new point is dominated by existing point. If yes, add.
    checker_new = all(diff'<0);
    if ~any(checker_new)
        y_PF = [y_PF; y_new];
        aux_PF = [aux_PF; aux_new];
    end
end

end
