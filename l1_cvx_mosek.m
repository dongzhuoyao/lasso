function [min_result_at_x,min_result] = l1_cvx_mosek(x0, A, b, mu, opts1)
n=size(A,1);
m=size(A,2);

cvx_begin
    cvx_solver mosek
    variable x(m)
    variable y(n)
    minimize 0.5*sum_square(y) + mu * norm(x, 1)
    subject to
    y == A * x - b
cvx_end

min_result_at_x=x;
min_result = cvx_optval;
end