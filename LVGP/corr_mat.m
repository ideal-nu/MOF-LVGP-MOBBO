function R = corr_mat(X1, X2, phi_full)

[~,d] = size(X1);

dist = 0;

for i = 1:d
    % distance matrix multiplied by theta
    dist = dist + (10^phi_full(i))*pdist2(X1(:,i),X2(:,i),'squaredeuclidean');
end

R = exp(-dist);
end


















