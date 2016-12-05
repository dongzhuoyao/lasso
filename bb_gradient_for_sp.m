function [min_result_at_x,min_result] = bb_gradient_for_sp(x0, A, b, mu, opts6)
n = size(A,2);
m = size(A,1);
%fprintf('iteration sub_gradient L2 norm Threshold: %f\n',sg_threshold)
%fprintf('iteration number: %i\n',iter_num)


myfun = @lasso;
[xst,  fst,  Gst, Out] = fminBB(x0, myfun, []);
%[xst,  Gst, Out] = fminGBB(x, myfun, []);

fprintf('\n\nxs \t \t xst \n');
fprintf('%4.3e \t %4.3e \n', [x0 xst]');
fprintf('||g||: %4.3e\n', norm(Gst));


Out

 function [f, g] = lasso(x)

f = 0.5*(norm(A*x-b,2)^2)+mu*norm(x,1);

g = zeros(n,1);
sign_value = arrayfun(@l1_smooth,x);
g = A'*(A*x-b)+mu*sign_value;
 end

 end


    
