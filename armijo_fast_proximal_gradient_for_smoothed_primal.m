function [min_result_at_x,min_result] = armijo_fast_proximal_gradient_for_smoothed_primal(x0, A, b, mu, opts6)
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
 
  new_y = pre_x-(i-2)*(pre_x-pre_pre_x)/(i+1);
  
  [step_size,aa] = steepdesc(new_y,@gx);
  delta_g = arrayfun(@smooth_gradient,new_y);
  u = new_y - step_size*delta_g;
  left = (eye(n)+step_size* A'*A)^-1;
  right1 = u;
  right2 = step_size*A'*b;
  right = right1+right2;
  new_x =  left * right;
  
  pre_pre_x = pre_x;
  pre_x =new_x;

  f_x = 0.5*(norm(A*new_x-b,2)^2)+mu*norm(new_x,1);
  
  min_result = f_x;
  min_result_at_x = new_x;
  fprintf('new minimal value: %f\n',min_result);
end


fprintf('min_value %f\n', min_result)
end

 function [f, g] = gx(x)
f = arrayfun(@smooth_fx,x);
g = arrayfun(@smooth_gradient,x);
 end


    
