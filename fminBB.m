function [x,  f,  g, Out] = fminBB(x, fun, opts, varargin)

% Barzilar Borwein Method with nonmonotone line search
%
%   min f(x),   g(x) = grad f(x)
%    s_{k-1} = x_k - x_{k-1};       y_{k-1} = g_k - g_{k-1}
%    BB step 1:     a = s_{k-1}^T y_{k-1} / || y_{k-1} ||^2
%    BB step 2:     a = ||s_{k-1}||^2 / s_{k-1}^T y_{k-1}
%
%
% Author: Zaiwen Wen
%   Version 1.0 .... 2010/10
%-------------------------------------------------------------------------

% choice of nonmontone line search scheme
if isfield(opts, 'method')
   if strcmp(opts.method, 'Raydan') ~=1 && strcmp(opts.method, 'HongChao') ~=1 
       opts.method = 'Raydan';
   end
else
     opts.method = 'Raydan';
end

% termination rule
if isfield(opts, 'xtol')
   if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-4;
   end
else
    opts.xtol = 1e-4;
end
    
if isfield(opts, 'gtol')
   if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-4;
   end
else
    opts.gtol = 1e-4;
end

% parameters for control the linear approximation in line search
if isfield(opts, 'rho')
   if opts.rho < 0 || opts.rho > 1
        opts.rho = 1e-4;
   end
else
    opts.rho = 1e-4;
end

% factor for decreasing the step size in the backtracking line search
if isfield(opts, 'eta')
   if opts.eta < 0 || opts.eta > 1
        opts.eta = 0.1;
   end
else
    opts.eta = 0.2;
end

% parameters for updating C by HongChao, Zhang
if isfield(opts, 'gamma')
   if opts.gamma < 0 || opts.gamma > 1
        opts.gamma = 0.85;
   end
else
    opts.gamma = 0.85;
end

% parameters for the  nonmontone line search by Raydan
if ~isfield(opts, 'M')
    opts.M = 10;
end

if ~isfield(opts, 'STPEPS')
    opts.STPEPS = 1e-10;
end

if isfield(opts, 'maxitr')
   if opts.maxitr < 0 || opts.maxitr > 2^20
        opts.maxitr = 1000;
   end
else
    opts.maxitr = 1000;
end

if ~isfield(opts, 'debug')
   opts.debug = 0;
end


%-------------------------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
gtol = opts.gtol;
rho  = opts.rho;
M     = opts.M;
STPEPS = opts.STPEPS;
eta   = opts.eta;
gamma = opts.gamma;


%% Initial function value and gradient

% prepare for iterations
[f,  g] = feval(fun, x , varargin{:});
Out.f = f;
Q = 1; C = f;   Out.nfe = 1;
gp = [];

nrmx = sum(sum(x.*x));
nrmG =  sqrt( sum(sum(g.*g)) );
alpha = 1e3;

% Print iteration header if debug == 1
if (opts.debug == 1)
    fid = 1;
    fprintf(fid, '\n----------- Projected Gradient Method with BB method ----------- \n');
    fprintf(fid, '%4s \t %10s \t %10s \t %10s \t %10s \t %10s\n', 'Iter', 'dt', 'f', 'CritdiffX', '||G||', 'StpBB' );
    fprintf(fid, '%4d \t %4.3e \t %4.3e \t %4.3e \t %4.3e \t %4.3e\n', 0, inf,f, inf, nrmG, 1/alpha);
    %OUTPUT
end

% main loop
for iter = 1:opts.maxitr

    % store old point
    xp = x;   nrmxp = nrmx; 
    gp = g;   %nrmGp = nrmG;
    
    % scale step size
    if alpha <= STPEPS || alpha >= 1/STPEPS
        if  nrmG > 1
            delta = 1;
        elseif nrmG >= 1e-5
            delta = 1/nrmG;
        else
            delta = 1e5;
        end
        alpha = delta;
    end
    stp = 1/alpha;

    %  nonmontone line search by Raydan

    deriv = rho*nrmG^2;
    
    nls = 1;
    while 1
        
        % calculate g, f,
        x = xp - stp*gp;
        [f,  g] = feval(fun, x, varargin{:});
        Out.nfe = Out.nfe + 1;
        
%         if (opts.debug == 1)
%             fprintf(fid, '\tLS: %4d \t stp:%4.3e \t C: %4.3e \t Cl: %4.3e \t f:%4.3e	\n', nls, stp, C, C-stp*deriv, f);
%         end
        
        if f <= C - stp*deriv || nls >= 20
            break
        end
        stp = eta*stp;
        nls = nls+1;
    end
  
    % s = x - xp = stp*gp;  ==> ||s|| = stp*||gp||
    nrms = stp*nrmG;
    % compute stopping 
    CritdiffX = nrms/max(nrmxp,1);
    
    % now, update normG
    nrmG =  sqrt( sum(sum(g.*g)) );
    Out.nrmg =  nrmG;
    Out.f = [Out.f; f];
    
    if (opts.debug == 1)
       fprintf(fid, '%4d \t %4.3e \t %4.3e \t %4.3e \t %4.3e \t %4.3e\n', iter, stp, f, CritdiffX, nrmG, 1/alpha);
    end
    
    if (CritdiffX < xtol) && (nrmG < gtol) 

        return;
    end
    

    % update for next iteration
    y = g - gp;
    gpTy = sum(sum(gp.*y));
    alpha = -gpTy/(stp*nrmG^2);
       
    % By Raydan
    if strcmp(opts.method, 'Raydan')
        C = max( Out.f( iter+1- min(iter, M): iter+1) );
    elseif strcmp(opts.method, 'Hongchao')
        % by HongChao Zhang
        Qp = Q; Q = gamma*Qp + 1; C = (gamma*Qp*C + f)/Q;
    end
    nrmx = sqrt(sum(sum(x.*x)));
    
end
Out.iter = iter;
