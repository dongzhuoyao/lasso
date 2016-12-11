function [min_result_at_x,min_result] = armijo_proximal_gradient_for_primal(x0, A, b, mu, opts6)
n = size(A,2);
gradient_threshold = opts6(1);
iter_num = opts6(2);
x = x0;
mu_target = mu;
mu = mu*1e5;

while mu>=mu_target
    
    for i= 1:iter_num
      gradient = A'*(A*x-b);
      fprintf('norm_gradient: %f\n', norm(gradient))
      if  norm(gradient) < gradient_threshold
          break;
      end
      %step_size = 4e-4;
      step_size = steepdesc(x,@gx,mu);
      u = x-step_size*gradient;
      
      %shrinkage 
      x = arrayfun(@shrinkage,u,mu*step_size*ones(n,1));    
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



    
