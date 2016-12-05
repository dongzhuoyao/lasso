function [min_result_at_x,min_result] = projection_gradient_descend_2(x0, A, b, mu, opts6)
n = size(A,2);
m = size(A,1);
sg_threshold = opts6(1);
iter_num = opts6(2);
min_result = 999999999999;
min_result_at_x = x0;
x1 = x0;
x2 = x0;
pre_gradient_x1=x1;
pre_gradient_x2=x2;


for i= 1:iter_num
  %fprintf('iteration %i\n', i)
  

  gradient_x1 = A'*(A*(x1-x2)-b)+mu;
  gradient_x2 = -A'*(A*(x1-x2)-b)+mu;
  
  fprintf('gradiant delta L2 norm: %f\n',norm(gradient_x1-pre_gradient_x1,2)+norm(gradient_x2-pre_gradient_x2,2));
  if norm(gradient_x1-pre_gradient_x1,2)+norm(gradient_x2-pre_gradient_x2,2)<sg_threshold
      %print 'projection L2 norm too small,stop...'
      break;
  end
  %{
  if i<1000
  step_size = 0.0001;
  else 
  step_size = 0.00000001;
  end
  %}
  
  g_k = gradient_x1-gradient_x2;
  z_k = A*g_k;
  step_size = (g_k'*g_k)/(z_k'*z_k);
  fprintf('step_size %f\n', step_size)
  
  x1 = x1-step_size*gradient_x1;
  x2 = x2-step_size*gradient_x2;
  
  %projection
  x1 = max(x1,0);
  x2 = max(x2,0);
  
  pre_gradient_x1 = gradient_x1;
  pre_gradient_x2 = gradient_x2;
  
  
  
  
  f_x = 0.5*(norm(A*(x1-x2)-b,2)^2)+mu*norm(x1-x2,1); 
  min_result = f_x;
  min_result_at_x = x1-x2;

fprintf('new minimal value: %f\n',min_result);
end


fprintf('min_value %f\n', min_result)
