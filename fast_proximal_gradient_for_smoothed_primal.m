function [min_result_at_x,min_result] = fast_proximal_gradient_for_smoothed_primal(x0, A, b, mu, opts6)
gradient_threshold = opts6(1);
iter_num = opts6(2);
pre_x = x0;
pre_pre_x = x0;
mu_target = mu;
mu = mu*1e3;
%continuation
while mu>=mu_target
        
    for i= 1:iter_num
      %sign_value = arrayfun(@my_sign,new_x);
      
      new_y = pre_x-(i-2)*(pre_x-pre_pre_x)/(i+1);
      
      delta_g = arrayfun(@l1_smooth,new_y);
      step_size = steepdesc(pre_x,@lasso,mu);
      
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
      
      [aa,sub_gradient] = lasso(pre_x,mu);
      step_size = steepdesc(pre_x,@lasso,mu);
      fprintf('step_size: %f\n',step_size);
      
      new_x = pre_x-step_size*sub_gradient;
      
      
      fprintf('subgradiant L2 norm: %f\n',norm(pre_x-new_x));
      if norm(pre_x-new_x)<sg_threshold
          fprintf('subgradient L2 norm too small,stop...\n');        
          break;
      end
      %update
      pre_x = new_x;
     
    end
    mu = mu/10.0;
end 
f_x = lasso(new_x,mu_target);
min_result = f_x;
min_result_at_x = new_x;
fprintf('new minimal value: %f\n',min_result);
fprintf('min_value %f\n', min_result)



    
