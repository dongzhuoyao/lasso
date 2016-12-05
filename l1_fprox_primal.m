function [x0, out] = l1_fprox_primal(x0, A, b, mu, opts)
%l1_fprox_primal Solving Lasso using fast proximal gradient method for the primal
%problem
%       min 0.5 ||Ax-b||_2^2 + mu*||x||_1
%   fast proximal gradient method for lasso:
%
%   x_{k+1} = argmin mu* ||x||_1 + g_k'(x-x_k) + 1/(2l) ||x - x_k||_ 2^2      
%           = shrink(x_k - l g_k, mu*l)
%   where shrink(y,l) = sign(y) .* max(|y| - l, 0)
maxit = 100;
steperr = 1e-7;
graderr = 1e-4;

l0 = 4e-4;

mu_t = mu;
%% fast proximal gradient method

% cached computations: there is no need!!!
% AtA = A'*A;
% Atb = A'*b;

% initialization of two key paras
mu =  1e5*mu;

% Continuation
while mu >= mu_t
   %% initialization
    x = x0;
    l = l0; % the step length
    k = 1;
    %% main loop
    while k < maxit 
        y = x + (k-2) / (k+1) * (x - x0);
        g = A'*(A*y-b);
        x0 = x;
        x_temp = y - l*g;   
        x = shrink(x_temp, mu*l);
        if norm(x-x0) < steperr || norm(g,inf) < graderr
            break
        end
        k = k+1;
    end
    mu = mu / 10;
end
out.x = x;
end

%%
%%%%%%%%%%%%%  Auxiliary functions  %%%%%%%%%%%%%
function y = shrink(x,l)
%shrinkage operator
y = sign(x) .* max(abs(x) - l, 0);
end