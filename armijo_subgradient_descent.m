 function [min_result_at_x,min_result] = armijo_subgradient_descent(x0, A, b, mu, opts6)
n = size(A,2);


sg_threshold = opts6(1);
iter_num = opts6(2);
%fprintf('iteration sub_gradient L2 norm Threshold: %f\n',sg_threshold)
%fprintf('iteration number: %i\n',iter_num)
min_result = 999999999999;
min_result_at_x = x0;
new_x = x0;
for i= 1:iter_num
  sign_value = arrayfun(@my_sign,new_x);
  sub_gradient = A'*(A*new_x-b)+mu*sign_value;
  fprintf('subgradiant L2 norm: %f\n',norm(sub_gradient,2));
  if norm(sub_gradient,2)<sg_threshold
      %print 'subgradient L2 norm too small,stop...'
      break;
  end
 

  step_size = steepdesc(new_x,@lasso);
  fprintf('step_size: %f\n',step_size);
  new_x = new_x-step_size*sub_gradient;
  f_x = 0.5*(norm(A*new_x-b,2)^2)+mu*norm(new_x,1);
  
  min_result = f_x;
  min_result_at_x = new_x;
  fprintf('new minimal value: %f\n',min_result);
end


fprintf('min_value %f\n', min_result)


 function [f, g] = lasso(x)
f = 0.5*(norm(A*x-b,2)^2)+mu*norm(x,1);
g = A'*(A*x-b)+mu*sign(x);
 end

 end




    
