function [min_result_at_x,min_result] = l1_gurobi(x0, A, b, mu, opts4)
clear model;
n = size(A,2);
m = size(A,1);
%{
names = {}
for xx=1:n
   names(end+1)={['name'+int2str(xx)]};
end
model.varnames = names;
%}

zero_n = zeros(n,n);
Q_up = [A'*A;zero_n];
Q_down=[zero_n;zero_n];
Q = [Q_up,Q_down];

model.Q = sparse(Q);

a_up = ones(1,2*n);
a_down = [-1*ones(1,n),ones(1,n)];
a_down_down = [zeros(1,n),ones(1,n)];
model.A = sparse([a_up;a_down;a_down_down]);

model.obj = [b'*A,mu*ones(1,n)];
model.rhs = zeros(1,3);
model.sense = '>';

results = gurobi(model);
%{
for v=1:length(names)
    fprintf('%s %e\n', names{v}, results.x(v));
end
%}
disp(results);
fprintf('Obj: %e\n', results.objval);
min_result=results.objval;
min_result_at_x = results.x(1:n);

