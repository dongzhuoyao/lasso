function [min_result_at_x,min_result] = armijo_fast_proximal_gradient_for_smoothed_primal(x0, A, b, mu, opts6)
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
      delta_g = A'*(A*y-b);
      fprintf('norm_gradient: %f\n', norm(delta_g))
      if  norm(delta_g) < gradient_threshold
          break;
      end
      %step_size = 4e-4;
      step_size = steepdesc(x,@gx,mu);
      u = y-step_size*delta_g;
      
      %shrinkage 
      x = arrayfun(@shrinkage,u,mu*step_size*ones(n,1)); 
      pre_pre_x = pre_x;
      pre_x = x;
    end
    
    mu = mu/10.0;
    
end


min_result_at_x = x;
min_result = 0.5*norm(A*x-b)^2+mu_target*norm(x,1);

fprintf('min_value %f\n', min_result)

function [f, g] = gx(x,mu)
   f = 0.5*norm(A*x-b)^2;
   g = A'*(A*x-b);
end

end



    
