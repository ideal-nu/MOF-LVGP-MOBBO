function [Y] = property_simulation_MOF(filename,X)
n = size(X,1);
data_s = table2array(readtable(filename));
Y_1 = data_s(:,1);
Y_2 = data_s(:,2);
X_s = data_s(:,3:end);
%X_s = orderMOF(X_s);
Y_1 = -Y_1;
Y_2 = -Y_2;
Y=[];
for i = 1:n
    idx = find(X(i,1) == X_s(:,1) &X(i,2) == X_s(:,2)& X(i,3) == X_s(:,3) &X(i,4) == X_s(:,4));
    Y(i,1) = Y_1(idx);Y(i,2)= Y_2(idx);
end
end

