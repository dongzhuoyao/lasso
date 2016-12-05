function [min_result_at_x,min_result] = proximal_gradient_for_primal(x0, A, b, mu, opts6)
gradient_threshold = opts6(1);
iter_num = opts6(2);
n = size(A,2);
m = size(A,1);
%fprintf('iteration sub_gradient L2 norm Threshold: %f\n',sg_threshold)
%fprintf('iteration number: %i\n',iter_num)
min_result = 999999999999;
min_result_at_x = x0;
new_x = [x0;x0];
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

  gradient_x1 = mu*ones(n,1);
  gradient_x2 = mu*ones(n,1);
  delta_g = [gradient_x1;gradient_x2];
  real_A = [A,-A;zeros(m,n),zeros(m,n)];
  real_b = [b;zeros(m,1)];
  
  u = new_x - step_size*delta_g;
  new_x = (eye(n*2)+step_size*real_A'*real_A)^-1 * (u+step_size*real_A'*real_b);

    % projection
  new_x = max(new_x,0);
  
  x1 = [new_x(1:n)];
  x2 = [new_x(n+1:2*n)];
  x = x1-x2;
  f_x = 0.5*(norm(A*x-b,2)^2)+mu*norm(x,1);
  
  min_result = f_x;
  min_result_at_x = new_x;
  fprintf('new minimal value: %f\n',min_result);
end


fprintf('min_value %f\n', min_result)
end


    
