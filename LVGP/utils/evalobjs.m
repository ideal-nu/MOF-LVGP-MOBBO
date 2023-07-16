%% update PF set with new point
function y_out = evalobjs(x, objs)

q = length(objs);
n = size(x,1);
y_out = zeros(n,2); % 2 objectives
for i = 1:n
    y_out(i,:) = objs(q).obj(x(i,:));
end


end
