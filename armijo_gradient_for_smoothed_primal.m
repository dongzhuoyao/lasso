function [min_result_at_x,min_result] = armijo_gradient_for_smoothed_primal(x0, A, b, mu, opts6)
n = size(A,2);
gradient_threshold = opts6(1);
iter_num = opts6(2);
new_x = x0;
mu_target = mu;
mu = mu*1e5;
while mu>=mu_target
    for i= 1:iter_num
      sign_value = arrayfun(@my_sign,new_x,0.002*ones(n,1));
      gradient = A'*(A*new_x-b)+mu*sign_value;
      fprintf('gradient L2 norm: %f\n',norm(gradient));
      step_size = steepdesc(new_x,@lasso,mu);
      %step_size = 1e-4;
      new_x = new_x-step_size*gradient;
      if norm(gradient)<gradient_threshold || norm(step_size*gradient)<1e-4
          %print 'gradient L2 norm too small,stop...'
          break;
      end
    end
    mu = mu/10.0;
end

f_x = 0.5*(norm(A*new_x-b,2)^2)+mu*norm(arrayfun(@smooth_fx,new_x),1);

min_result = f_x;
min_result_at_x = new_x;
fprintf('new minimal value: %f\n',min_result);



function [f, g] = lasso(x,mu)
f = 0.5*(norm(A*x-b,2)^2)+mu*norm(arrayfun(@smooth_fx,x),1);
g = A'*(A*x-b)+mu*arrayfun(@smooth_gradient,x);
 end

 end
