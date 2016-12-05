function [min_result_at_x,min_result] = armijo_proximal_gradient_for_primal(x0, A, b, mu, opts6)
n = size(A,2);
sg_threshold = opts6(1);
iter_num = opts6(2);
min_result = 999999999999;
min_result_at_x = x0;
x = [x0;x0];


for i= 1:iter_num
  [aa,gradient] = gx(x);  
  fprintf('gradiant  L2 norm: %f\n',norm(gradient));
  if norm(gradient)<sg_threshold
      %print 'projection L2 norm too small,stop...'
      break;
  end
  step_size = steepdesc(x,@gx);
  fprintf('step_size %f\n', step_size)
  
  u = x-step_size*gradient;
  real_A = A*[eye(n),-1*eye(n)];
  left = (eye(2*n)+step_size* real_A'*real_A)^-1;
  right1 = u;
  right2 = step_size*real_A'*b;
  right = right1+right2;
  x =  left * right;

  %projection
  x = max(x,0);
  
  
  [f_x,g_x] = lasso(x) ; 
  min_result = f_x;
  min_result_at_x = x;

fprintf('new minimal value: %f\n',min_result);
end


fprintf('min_value %f\n', min_result)

function [f, g] = gx(x)

f = mu*sum(sum(x));
g =mu*ones(2*n,1);
 end

end



    
