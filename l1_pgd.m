function [min_result_at_x,min_result] = l1_pgd(x0, A, b, mu, opts6)
n = size(A,2);
m = size(A,1);
sg_threshold = opts6(1);
iter_num = opts6(2);
min_result = 999999999999;
min_result_at_x = x0;
new_x = x0;
for i= 1:iter_num
  %fprintf('iteration %i\n', i)
  init_x = new_x;

  sign_value = arrayfun(@my_sign,init_x);
  sub_gradient = A'*(A*init_x-b)+mu*sign_value;
  
  fprintf('subgradiant L2 norm: %f\n',norm(sub_gradient,2));
  if norm(sub_gradient,2)<sg_threshold
      %print 'projection L2 norm too small,stop...'
      break;
  end
  step_size = 0.004/sqrt(i);

  new_x = init_x-step_size*sub_gradient;
  
  %projection
  new_x = max(new_x,0);
  
  f_x = sum(square(A*new_x-b))+mu*sum(abs(new_x));
  
  min_result = f_x;
  %fprintf('new minimal value: %f\n',min_result);
  min_result_at_x = new_x;
end


%fprintf('min_value_at \n')
%disp(min_result_at_x)

fprintf('min_value %f\n', min_result)
