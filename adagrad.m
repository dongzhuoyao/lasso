 function [min_result_at_x,min_result] = adagrad(x0, A, b, mu, opts6)

sg_threshold = opts6(1);
iter_num = opts6(2);
pre_x = x0;
mu_target = mu;
mu = mu;
%continuation
while mu>=mu_target
        
    for i= 1:iter_num
      %sign_value = arrayfun(@my_sign,new_x,0.02*one(n,1));
      [aa,sub_gradient] = lasso(pre_x,mu);
      fprintf('norm gradient: %f\n',norm(sub_gradient)); 
       if norm(sub_gradient)<sg_threshold
          fprintf('subgradient L2 norm too small,stop...\n');        
          break;
       end
      
      step_size = 0.0001;
      grad_square_sum = sum(sub_gradient);
      step_size = step_size/(sqrt(grad_square_sum)+1e-7);
      fprintf('step_size: %f\n',step_size); 
      new_x = pre_x-step_size*sub_gradient;  
     

      %update
      pre_x = new_x;
     
    end
    mu = mu/10.0;
end 
f_x = lasso(new_x,mu_target);
min_result = f_x;
min_result_at_x = new_x;

fprintf('min_value %f\n', min_result)


function [f, g] = lasso(x,mu)
f = 0.5*(norm(A*x-b)^2)+mu*norm(x,1);
g = A'*(A*x-b)+mu*sign(x);
 end

 end




    
