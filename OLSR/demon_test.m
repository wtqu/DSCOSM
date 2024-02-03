clc; close all; clear all;
n            = 100;                                         % the number of samples
m            = 12800;                                       % the number of features
p            = 10;                                          % the number of classes
s            = ceil(0.05*n);                                % the sparsity
[A,B,X_ture] = CoReData(n,m,p,s);                           % Generate data

data.A = A;
data.B = B;
data.X_ture = X_ture;


[n,m] = size(data.A);
options.maxiter   = 100;
options.num_block = 16;
options.s         = s;
options.tol       = 1e-5;
options.mu        = 2^(4);
options.beta      = 1;


out_DSCOSM  =  DSCOSM_OLSR(data, options);
RMSE_DSCOSM = 0;
for i = 1:length(out_DSCOSM.idx_block)
    Ai = A(:,out_DSCOSM.idx_block{i})';
    [~,mi] = size(Ai);
    RMSE_DSCOSM = RMSE_DSCOSM + sqrt(norm(Ai*out_DSCOSM.Y-B,'fro')^2/(mi*p));
end

