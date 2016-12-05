function [min_result_at_x,min_result] = fast_proximal_gradient_for_smoothed_primal(x0, A, b, mu, opts6)
gradient_threshold = opts6(1);
iter_num = opts6(2);
n = size(A,2);
m = size(A,1);
%fprintf('iteration sub_gradient L2 norm Threshold: %f\n',sg_threshold)
%fprintf('iteration number: %i\n',iter_num)
min_result = 999999999999;
min_result_at_x = x0;
pre_x = x0;
pre_pre_x = x0;
for i= 1:iter_num
  %fprintf('iteration %i\n', i)
  
  %{
  step_size = 0.004/sqrt(i);
  %}
  if i<1000
  step_size = 0.01;
  else 
  step_size = 0.01;
  end

  new_y = pre_x-(i-2)*(pre_x-pre_pre_x)/(i+1);
  
  delta_g = arrayfun(@l1_smooth,new_y);
  u = new_y - step_size*delta_g;
  left = (eye(n)+step_size* A'*A)^-1;
  right1 = u;
  right2 = step_size*A'*b;
  right = right1+right2;
  new_x =  left * right ;
  
  pre_pre_x = pre_x;
  pre_x =new_x;

  f_x = 0.5*(norm(A*new_x-b,2)^2)+mu*norm(new_x,1);
  
  min_result = f_x;
  min_result_at_x = new_x;
  fprintf('new minimal value: %f\n',min_result);
end


fprintf('min_value %f\n', min_result)
end


    
