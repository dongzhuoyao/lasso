% function Test_l1_regularized_problems

% min 0.5 ||Ax-b||_2^2 + mu*||x||_1

% generate data
rng('default');
rng(3);
n = 1024;
m = 512;

A = randn(m,n);
u = sprandn(n,1,0.1);
b = A*u;

mu = 1e-3;

x0 = rand(n,1);

errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));

% cvx calling mosek
opts1 = []; %modify options
tic; 
[x1, out1] = l1_cvx_mosek(x0, A, b, mu, opts1);
t1 = toc;


opts11 = [1e-4,100]; %modify options
tic; 
[x11, out11] = armijo_fast_proximal_gradient_for_smoothed_primal(x0, A, b, mu, opts11);
t11 = toc;

opts7 = [1e-4,100]; %modify options
tic; 
[x7, out7] = armijo_gradient_for_smoothed_primal(x0, A, b, mu, opts7);
t7 = toc;

opts5 = [1e-4,100]; %modify options
tic; 
[x5, out5] = armijo_projection_gradient_descend_2(x0, A, b, mu, opts5);
t5 = toc;

opts6 = [1e-4,100]; %modify options
tic; 
[x6, out6] = armijo_subgradient_descent(x0, A, b, mu, opts6);
t6 = toc;

opts9 = [1e-4,100]; %modify options
tic; 
[x9, out9] = armijo_proximal_gradient_for_primal(x0, A, b, mu, opts9);
t9 = toc;




% cvx calling gurobi
opts2 = []; %modify options
tic; 
[x2, out2] = l1_cvx_gurobi(x0, A, b, mu, opts2);
t2 = toc;



%{
SLOW

%}







%{ 
 TODO




%}



opts10 = [0.05,10000]; %modify options
tic; 
[x10, out10] = proximal_gradient_for_primal(x0, A, b, mu, opts10);
t10 = toc;





opts8 = [0.05,10000]; %modify options
tic; 
[x8, out8] = l1_fast_gradient_for_sp(x0, A, b, mu, opts8);
t8 = toc;




%{

% call mosek directly
opts3 = []; %modify options
tic; 
[x3, out3] = l1_mosek(x0, A, b, mu, opts3);
t3 = toc;

% call gurobi directly
opts4 = []; %modify options
tic; 
[x4, out4] = l1_gurobi(x0, A, b, mu, opts4);
t4 = toc;

% other approaches
%}



% print comparison results with cvx-call-mosek
%{
fprintf('call-mosek: cpu: %5.2f, err-to-cvx-mosek: %3.2e, optimal:%5.3f\n', t3, errfun(x1, x3),out3);
fprintf('call-gurobi: cpu: %5.2f, err-to-cvx-mosek: %3.2e, optimal:%5.3f\n', t4, errfun(x1, x4),out4);
%}
fprintf('cvx-call-gurobi: cpu: %5.2f, err-to-cvx-mosek: %3.2e, optimal:%5.3f\n', t2, errfun(x1, x2),out2);


fprintf('projection gradient method: cpu: %5.2f, err-to-cvx-mosek: %3.2e, optimal:%5.3f\n', t5, errfun(x1, x5),out5);
fprintf('subgradient method: cpu: %5.2f, err-to-cvx-mosek: %3.2e, optimal:%5.3f\n', t6, errfun(x1, x6),out6);
fprintf('subgradient method: cpu: %5.2f, err-to-cvx-mosek: %3.2e, optimal:%5.3f\n', t7, errfun(x1, x7),out7);
