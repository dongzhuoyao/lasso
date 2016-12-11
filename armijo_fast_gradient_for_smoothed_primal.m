function [min_result_at_x,min_result] = armijo_fast_gradient_for_smoothed_primal(x0, A, b, mu, opts6)
n = size(A,2);
gradient_threshold = opts6(1);
iter_num = opts6(2);
x = x0;
pre_x = x0;
pre_pre_x =x0;

mu_target = mu;
mu = mu*1e5;

while mu>=mu_target
    
    for i= 1:iter_num
      y = pre_x+(i-2)*(pre_x-pre_pre_x)*1.0/(i+1); 
      delta_g = A'*(A*y-b)+mu*arrayfun(@smooth_gradient,y);
      fprintf('norm_gradient: %f\n', norm(delta_g))
      if  norm(delta_g) < gradient_threshold
          break;
      end
      step_size = 1e-4;
      %step_size = steepdesc(x,@lasso,mu); too slow!!
      x = y-step_size*delta_g;
      
      pre_pre_x = x;
      pre_x =x;
      
      
    end
    
    mu = mu/10.0;
    
end


min_result_at_x = x;
min_result = 0.5*norm(A*x-b)^2+mu_target*norm(x,1);

fprintf('min_value %f\n', min_result)

function [f, g] = lasso(x,mu)
f = 0.5*(norm(A*x-b,2)^2)+mu*arrayfun(@smooth_fx,x);
g = A'*(A*x-b)+mu*arrayfun(@smooth_gradient,x);
 end


end



    
