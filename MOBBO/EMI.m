function [EMI_val, meanpred, MSEpred] = EMI(UQmodels, predfunc, predoptions, x_can, y_PF)
%{
This function calculates the Expected-Maximin Improvement (EMI) for a given
design candidate
Input variables:
UQmodels = struct, LVGP models
predfunc = MATLAB function, LVGP predictor
predoptions = struct, LVGP predictor options
x_can = ?-p matrix, Design candidates
y_PF = ?-by-q matrix, Current Pareto Front properties

Output variables:
EMI_val = 1x1 float, MOEI value of candidate design
meanpred = ?-by-q matrix, Predicted objectives of candidate design
msepred = ?-by-q matrix, Predicted variance of candidate designs
%}

q = length(UQmodels);
n_PF = size(y_PF,1);
meanpred = zeros(1, q);
MSEpred = zeros(1, q);
for i = 1:q
    output = predfunc(x_can, UQmodels(i).model, predoptions);
    meanpred(i)=output.Y_hat;
    MSEpred(i) =output.MSE;
end
EI_1 = zeros(n_PF,1); 
EI_2 = zeros(n_PF,1);
s = sqrt(abs(MSEpred));
for p = 1:n_PF
    b = (y_PF(p,:) - meanpred)./s;
    EI_1(p,1) = real((y_PF(p,1)-meanpred(1)).*normcdf(b(:,1)) + s(1).*normpdf(b(:,1)));
    EI_2(p,1) = real((y_PF(p,2)-meanpred(2)).*normcdf(b(:,2)) + s(2).*normpdf(b(:,2)));
end
EMI_val = min(max([EI_1,EI_2],[],2));
end






