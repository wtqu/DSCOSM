function [out] = DSCOSM_OLSR(data, options)
%This code aims at solving the DSCOSM for OLSR
%     \min_{Y,X_i} \sum_{i=1}^d ||A_i^T*X_i-B_i^T||_F^2 + mu/4*||Y'Y-I_p||_F^2
%     s.t. ||Y||_{2,0} <=s,
% where Y, X_i\in\R^{n\times p}, s<<n is an given integer
%       ||Y||_{2,0} is the number of nonzero rows of Pv
%
% Inputs: 
%        A\in R^{n \times m}: m and n are the number of samples and features, respectively
%        B\in R^{p \times m}  
%        options  -- a structure
%                 options.s         -- the sparsity level, an integer in (0,n)      (required)
%                 options.maxiter   -- the maximum number of iterations
%                 options.mu        -- the tuning parameters
%                 options.num_block -- the number of agents
%                 options.beta      -- the penalty parameter
%                 options.tol       -- the default stopping error
%
% Outputs: 
%       out.Y           -- the sparse solution
%       out.X           -- the auxiliary variables
%       out.idx_block   -- the label for sample segmentation
%       out.obj         -- objective function values
%       out.iter        -- number of iterations
%       out.Error_Y     -- error1
%       out.Error_obj   -- error2
%       out.time        -- runing time

%% get data sequence
A = data.A;
B = data.B;

%% Initialization
if isfield(options,'maxiter');   maxiter   = options.maxiter;   else; maxiter   = 1e3;  end
if isfield(options,'beta');      beta      = options.beta;      else; beta      = 1e3;  end
if isfield(options,'mu');        mu        = options.mu;        else; mu        = 2^5;  end
if isfield(options,'num_block'); num_block = options.num_block; else; num_block = 10;   end
if isfield(options,'s');         s         = options.s;         else; s         = 10;  end
if isfield(options,'tol');       tol       = options.tol;       else; tol       = 1e-5; end
eta   = 1e-4;
sigma = 1e-8;
[n,m] = size(A);
[p,~] = size(B);
Ip = eye(p); In = eye(n); Is = eye(s);

X0      = normrnd(0,1,[n,p]);
Y0      = normrnd(0,1,[n,p]);
Lambda0 = zeros(n,p);

%% Defining Functions
f_loss = @(X,A,B) (norm(A'*X-B','fro'))^2/(2*n);
g_loss = @(X) (mu/4*(norm(X'*X-Ip,'fro'))^2);
Fnorm   = @(var)norm(var,'fro')^2;

%% assign data
idx_data_shuffled = randperm(m); % randomly permute the data for assignment to blocks
size_data_block = floor(m/num_block); X = {}; Lambda = {};
for i = 1:num_block
    idx_block{i} = idx_data_shuffled(((i-1)*size_data_block+1):(i*size_data_block));
    A_block{i} = A(:,idx_block{i});         B_block{i} = B(:,idx_block{i});    
    X{i} = X0;                              Lambda{i} = Lambda0;
end

Y_current = Y0;

fprintf(' Run solver DSMNAL---------------------------------%5d\n',num_block);
fprintf('Iter\t  ObjVal\t error_obj\t  error_Y\n');

tic;
for iter = 1:maxiter
    [h_loss, h_grad] = funch(Y_current,X,Lambda,num_block,beta,mu,Ip);                            % Calculate the gradient and function values of h
    [~,Tu]           = maxk( sum((Y_current - eta * h_grad).^2,2),s,'ComparisonMethod','abs');    % Find the support indices Tu
    YT               = Y_current(Tu,:);        
    
    %% update Y by NHTP
    for initer = 1 : 4
        Y_old      = Y_current; 
        h_grad_YT  = h_grad(Tu,:);                                                      % Compute (\nabla h(Y))_Tu
        err_newton = Fnorm(h_grad_YT);
%         fprintf('initer = %5d\t  err_newton = %6.2e\t\n',initer, err_newton);
        if err_newton < tol; break; end                                                 % check the stop criteria of NHTP

        Kron  = kron(Ip,YT* (YT')); 
        Kron2 = kron((YT')*YT,Is); 
        Kron3 = kron((YT'),YT); 
        nb_beta_mu = num_block * beta- mu;
        H_grad_1   = mu * (Kron+Kron2+ Kron3)+ nb_beta_mu* eye(s*p);                    % calculate the hessian of h(Y)
        
        vec_h_grad_YT = reshape(h_grad_YT,[],1);      
        vec_D_Tu = (H_grad_1)\(-vec_h_grad_YT );                                        % solve the Newton equation   

        D_Tu   = reshape(vec_D_Tu,[s,p]);                                               % update the search direction D
        temp1  = max(0,Fnorm(Y_current)-Fnorm(YT)); 
        marker = (trace(D_Tu'*h_grad_YT) >= -(1e-10)*(Fnorm(D_Tu)) + temp1/4/eta );
        if Fnorm(D_Tu) > 1e16 || marker
            D_Tu = - h_grad_YT;
        end
        D         = -Y_current;
        D(Tu,:)   = D_Tu;

        % Armijio line search
        alpha     = 1;
        Y_current = zeros(n,p);
        temp2     = sigma * trace(h_grad'*D);  
        for mm = 1:10
            Y_current(Tu,:)      = Y_old(Tu,:) + alpha * D_Tu;
            [h_loss_new, h_grad] = funch(Y_current,X,Lambda,num_block,beta,mu,Ip);
            if h_loss_new <= (h_loss+alpha*temp2);  break; end
            alpha = alpha/2;
        end

        [~,Tu] = maxk( sum((Y_current - eta * h_grad).^2,2),s,'ComparisonMethod','abs'); % find the support indices Tu
        YT     = Y_current(Tu,:);   
        if mod(initer,5)==0; eta = max(eta/2,1e-5); end
    end
    
    %% update X and Lambda
    f_cost = 0;
    for j = 1:num_block
        A_local = A_block{j};	B_local = B_block{j};	Lambda_local = Lambda{j};
        X{j} = pinv(A_local*(A_local')+beta* In)*(A_local*(B_local') + beta*Y_current + Lambda_local);      % update X
        Lambda{j} = Lambda_local - beta*(X{j}-Y_current);               % update Lambda
        f_cost = f_cost+ f_loss(X{j},A_local,B_local)/(2*num_block);   
    end
    obj(iter) = f_cost + g_loss(Y_current);                                     % calculate the loss function
    
    %%  check the stop criteria
    if iter>1
        error_obj(iter) = abs(obj(iter)-obj(iter-1))/(1+abs(obj(iter-1)));      % error_obj
        error_Y(iter)   = Fnorm(Y_current-Y_old)/(1+Fnorm(Y_old));              % error_Y
        fprintf('%5d\t  %6.2e\t %6.2e\t  %6.2e\n',iter, obj(iter), error_obj(iter), error_Y(iter));
        if (max([error_obj(iter),error_Y(iter)]) <tol); break; end
        if iter == maxiter; fprintf('The number of iterations reaches maxiter.\n'); end
    end
end
time = toc;
out.Y           = Y_current;
out.X           = X;
out.idx_block   = idx_block;
out.obj         = obj;
out.iter        = iter;
out.Error_Y     = error_Y;
out.Error_obj   = error_obj;
out.time        = time;
end

function [h_loss, h_grad] = funch(Y,X,Lambda,num_block,beta,mu,Ip)
%% Calculate the gradient and function values of h
    gh_grad  = 0; h_loss_2 = 0;
    for i = 1:num_block                                                                 %% Calculate the gradient and function values of h
        X_local  = X{i};     Lambda_local = Lambda{i}; 
        gh_grad  = gh_grad - beta*(X_local-Y-Lambda_local/beta);
        h_loss_2 = h_loss_2 + beta/2*(norm(X_local-Y-Lambda_local/beta,'fro')^2); 
    end
    h_grad = mu*Y*((Y')*Y-Ip) + gh_grad;                                              % gradient of h
    h_loss = mu/4*(norm(Y'*Y-Ip,'fro'))^2 + h_loss_2/2;                                            % function values of h
end