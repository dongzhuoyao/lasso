function [min_result_at_x,min_result] = l1_mosek(x0, A, b, mu, opts3)
clear prob;
n = size(A,2);
m = size(A,1);
tmp = mu*ones(1,n);
b_trans_plus_A = b'*A;
c = [b_trans_plus_A,tmp]';
prob.c = c;

% Define the data.

% First the lower triangular part of q in the objective 
% is specified in a sparse format. The format is:
%
%   Q(prob.qosubi(t),prob.qosubj(t)) = prob.qoval(t), t=1,...,4
zero_n = zeros(n,n);
Q_up = [A'*A;zero_n];
Q_down=[zero_n;zero_n];
Q = [Q_up,Q_down];

prob.qosubi=[];
prob.qosubj=[];
prob.qoval=[];
for i = 1:size(Q, 1)
   for j = 1:i
       prob.qosubi(end+1)=i;
       prob.qosubj(end+1) = j;
       prob.qoval(end+1) = Q(i,j); 
   end
end

% a, the constraint matrix
subi  = ones(1,2*n);
subj  = [-1*ones(1,n),ones(1,n)];
subk = [zeros(1,n),ones(1,n)];
prob.a = [subi;subj;subk];

% Lower bounds of constraints.
prob.blc  = [0,0,0]';

% Upper bounds of constraints.
prob.buc  = [inf,inf,inf]';

% Lower bounds of variables.
blx = [-inf*ones(1,n),zeros(1,n)];
prob.blx  = blx;

% Upper bounds of variables.
prob.bux = [];   % There are no bounds.

[r,res] = mosekopt('minimize',prob);

% Display return code.
fprintf('Return code: %d\n',r);

% Display primal solution for the constraints.
res.sol.itr.xc;

% Display primal solution for the variables.
res.sol.itr.xx;
min_result_at_x = res.sol.itr.xx(1:n);
min_result=sum(square(A*res.sol.itr.xx(1:n)-b))+mu*sum(abs(res.sol.itr.xx(1:n)));


