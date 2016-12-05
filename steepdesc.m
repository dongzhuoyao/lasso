function [step_size,  nn] = steepdesc(x, func)
% Armijo stepsize rule parameters
  sigma = .1;
  beta = .8;
  [obj,g] = feval(func, x );
  nf=1;		
  
    d = -g;                   % steepest descent direction
    a = 1;
    newobj = func(x + a*d);
    while (newobj-obj)/a > sigma*g'*d
      a = a*beta;
      newobj = func(x + a*d);
      nf = nf+1;
    end 
% Output a and nf
  step_size = a;
  nn =nf;