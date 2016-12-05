function [step_size,  nn] = steepdesc(x, func,mu)
% Armijo stepsize rule parameters
  sigma = .1;
  beta = .5;
  [obj,g] = feval(func, x,mu);
  nf=1;		
  
    d = -g;                   % steepest descent direction
    a = 1;
    newobj = func(x + a*d,mu);
    while (newobj-obj)/a > sigma*g'*d
      a = a*beta;
      newobj = func(x + a*d,mu);
      nf = nf+1;
    end 
% Output a and nf
  step_size = a;
  nn =nf;