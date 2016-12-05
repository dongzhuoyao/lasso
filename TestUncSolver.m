function TestUncSolver

x = [ 0;0];

x = [ -1.2;1];

myfun = @rosenbr;
% myfun = @polyfun;

% [xst,  fst,  Gst, Out] = fminSteep(x, myfun, []);
% 
% fprintf('xs \t \t xst \n');
% fprintf('%4.3e \t %4.3e \n', [x xst]');
% fprintf('||g||: %4.3e\n', norm(Gst));

[xst,  fst,  Gst, Out] = fminBB(x, myfun, []);
%[xst,  Gst, Out] = fminGBB(x, myfun, []);

fprintf('\n\nxs \t \t xst \n');
fprintf('%4.3e \t %4.3e \n', [x xst]');
fprintf('||g||: %4.3e\n', norm(Gst));
Out


opts.debug = 0;
opts.gtol = 1e-5;
% [xst,  fst,  Gst, Out] = fminLBFGS_s(x, myfun, opts);
[xst,  fst,  Gst, Out] = fminLBFGS_Loop(x, myfun, opts);
% [xst,  fst,  Gst, Out] = fminLBFGSBB(x, myfun, opts);

fprintf('\n\nxs \t \t xst \n');
fprintf('%4.3e \t %4.3e \n', [x xst]');
fprintf('||g||: %4.3e\n', norm(Gst));
Out

function [f, g] = rosenbr(x)

f = 100*(x(2)-x(1)^2)^2+(x(1)-1)^2;

g = zeros(2,1);
g(1) = 2*(x(1)-1) - 400*x(1) *(x(2) - x(1)^2);
g(2) = 200*(x(2) - x(1)^2);


function [f,g] = polyfun(x)
f = 3*x(1)^2 + 2*x(1)*x(2) + x(2)^2;    % Cost function
if nargout > 1
   g(1,1) = 6*x(1)+2*x(2);
   g(2,1) = 2*x(1)+2*x(2);
end
