 function [min_result_at_x,min_result] = admm_linearization_for_primal(x0, A, b, mu, opts6)
sg_threshold = opts6(1);
m = size(A,1);
iter_num = opts6(2);
c = 1e-4;
beta = 1e-4;

x = x0;
z = x0(1:m);
y = x0(1:m);

mu_target = mu;
mu = mu*1e5;

%continuation
while mu>=mu_target
        
    for i= 1:iter_num
      x = x - c*(arrayfun(@smooth_gradient,x)+beta*A'*(A*x-z-b-y));
      fprintf('z %f\n', z);
      z = z - c*(z+0.5*beta*(A*x-z-b-y));
      y = y - (A*x-z-b);  
    end
    mu = mu/10.0;
end 
f_x = lasso(x,mu_target);
min_result = f_x;
min_result_at_x = x;
fprintf('new minimal value: %f\n',min_result);
fprintf('min_value %f\n', min_result)


 function [f, g] = lasso(x,mu)
f = 0.5*(norm(A*x-b)^2)+mu*norm(x,1);
g = A'*(A*x-b)+mu*sign(x);
 end

 end




    
