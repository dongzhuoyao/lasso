 function [min_result_at_x,min_result] = subgradient_descent(x0, A, b, mu, opts6)
sg_threshold = opts6(1);
iter_num = opts6(2);
%fprintf('iteration sub_gradient L2 norm Threshold: %f\n',sg_threshold)
%fprintf('iteration number: %i\n',iter_num)
min_result = 999999999999;
min_result_at_x = x0;
new_x = x0;
for i= 1:iter_num
  %fprintf('iteration %i\n', i)
 
  sign_value = arrayfun(@my_sign,new_x);
  %{
  if i<500
    mu = 1000;
  elseif i<700
    mu = 100;
 elseif i<900
    mu = 10;   
      elseif i<1100
    mu = 1;   
      elseif i<1300
    mu=0.1;    
      elseif i<1600
  mu = 0.01;
  else
     mu =0.001;
  end
  %}
  sub_gradient = A'*(A*new_x-b)+mu*sign_value;
  fprintf('subgradiant L2 norm: %f\n',norm(sub_gradient,2));
  if norm(sub_gradient,2)<sg_threshold
      %print 'subgradient L2 norm too small,stop...'
      break;
  end
  %{
  step_size = 0.004/sqrt(i);
  %}
  if i<1000
  step_size = 0.0001;
  else 
  step_size = 0.000001;
  end

  new_x = new_x-step_size*sub_gradient;
  f_x = square(norm(A*new_x-b,2))+mu*norm(new_x,1);
  
  min_result = f_x;
  min_result_at_x = new_x;
  fprintf('new minimal value: %f\n',min_result);
end


fprintf('min_value %f\n', min_result)
end


    
