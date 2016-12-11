function [min_result_at_x,min_result] = armijo_projection_gradient_descend_2(x0, A, b, mu, opts6)
n = size(A,2);
sg_threshold = opts6(1);
iter_num = opts6(2);
min_result = 999999999999;
x = [x0;x0];
mu_target = mu;
mu = mu*1e5;

while mu>=mu_target
    
  for i= 1:iter_num
      [aa,gradient] = lasso(x,mu);  
      fprintf('gradiant  L2 norm: %f\n',norm(gradient));
      if norm(gradient)<sg_threshold
          %print 'projection L2 norm too small,stop...'
          break;
      end
      %step_size = steepdesc(x,@lasso,mu);
      step_size = 1e-4;
      x = x-step_size*gradient;
      %projection
      x = max(x,0);
  end
  mu = mu/10.0;
end
[f_x,g_x] = lasso(x,mu_target) ; 
min_result = f_x;
min_result_at_x = x;
fprintf('new minimal value: %f\n',min_result);
fprintf('min_value %f\n', min_result)

function [f, g] = lasso(x,mu)
f = 0.5*(norm(A*[eye(n),-1*eye(n)]*x-b,2)^2)+mu*sum(sum(x));
g1 = A'*(A*[eye(n),-1*eye(n)]*x-b)+mu*ones(n,1);
g2 = -A'*(A*[eye(n),-1*eye(n)]*x-b)+mu*ones(n,1);
g =[g1;g2];
 end

end
