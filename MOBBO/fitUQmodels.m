%% update PF set with new point
function models = fitUQmodels(x, y, fitfunc, fitoptions)

q = size(y, 2);
models = struct('model',cell(q,1));
for i = 1:q
    models(i).model = fitfunc(x, y(:,i), fitoptions);
end


end