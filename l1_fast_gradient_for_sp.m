function [min_result_at_x,min_result] = l1_fast_gradient_for_sp(x0, A, b, mu, opts6)
sg_threshold = opts6(1);
iter_num = opts6(2);
n = size(A,2);
m = size(A,1);
%fprintf('iteration sub_gradient L2 norm Threshold: %f\n',sg_threshold)
%fprintf('iteration number: %i\n',iter_num)
min_result = 999999999999;
min_result_at_x = x0;
new_x = x0;
pre_x = x0;
pre_pre_x = x0;
for i= 1:iter_num
  %fprintf('iteration %i\n', i)
  
  %{
  step_size = 0.004/sqrt(i);
  %}
  if i<1000
  step_size = 0.0001;
  else 
  step_size = 0.000001;
  end

  new_y = pre_x-(i-2)*(pre_x-pre_pre_x)/(i+1);
  delta_g = arrayfun(@l1_smooth,new_y);
  u = new_y - step_size*delta_g;
  
  
  new_x = (eye(n)+step_size*A'*A)^-1 * (u+step_size*A'*b);
  pre_pre_x = pre_x;
  pre_x = new_x;
  
  f_x = square(norm(A*new_x-b,2))+mu*norm(new_x,1);
  
  min_result = f_x;
  min_result_at_x = new_x;
  fprintf('new minimal value: %f\n',min_result);
end


fprintf('min_value %f\n', min_result)
end


    
