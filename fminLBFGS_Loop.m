function [x, f, g, Out, CritdiffX]= fminLBFGS(x, fun, opts, varargin)

% Limited Memory BFGS Method
%
% Author: Zaiwen Wen
%   Version 1.0 .... 2010/10
%-------------------------------------------------------------------------

n = length(x);
% termination rule
if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-6;
    end
else
    opts.xtol = 1e-6;
end

if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-6;
    end
else
    opts.gtol = 1e-6;
end

if isfield(opts, 'm')
    if opts.m < 0 || opts.m > n
        opts.m = 5;
    end
else
    opts.m = 5;
end

% parameters for control the linear approximation in line search
if isfield(opts, 'rho1')
    if opts.rho1 < 0 || opts.rho1 > 0.5
        opts.rho1 = 1e-4;
    end
else
    opts.rho1 = 1e-4;
end
parsls.ftol = opts.rho1;

% parameters for control the linear approximation in line search
if isfield(opts, 'rho2')
    if opts.rho2 < opts.rho1 || opts.rho2 > 1
        opts.rho2 = 0.9;
    end
else
    opts.rho2 = 0.9;
end
parsls.gtol = opts.rho2;

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
m = opts.m;


%% Initial function value and gradient
% prepare for iterations
[f,  g] = feval(fun, x , varargin{:});
Out.f = [];  Out.nfe = 1;

nrmx = norm(x);
nrmG = norm(g);

% set up storage for L-BFGS
SK = zeros(n,m);		  % S stores the last ml changes in x
YK = zeros(n,m);		  % Y stores the last ml changes in gradient of f.
istore = 0; pos = 0;  status = 0;  perm = [];

% Print iteration header if debug == 1
if (opts.debug >= 1)
    fid = 1;
    fprintf(fid, '\n----------- BFGS Method with CSRCH Line Search ----------- \n');
    fprintf(fid, '%4s \t %10s \t %10s \t %10s \t %10s\n', 'Iter', 'dt', 'f', 'CritdiffX', '||G||' );
    fprintf(fid, '%4d \t %4.3e \t %4.3e \t %4.3e \t %4.3e\n', 0, inf, f, inf, nrmG);
    %OUTPUT
end

% main loop
for iter = 1:opts.maxitr

    % begin line search
    % clear workls;

    % store old point
    xp = x;   nrmxp = nrmx;
    fp = f;   gp = g;   %nrmGp = nrmG;

    % compute search direction
    % if the first iteration
    if istore == 0
        d = -g;
    else
        d = LBFGS_Hg_Loop(-g);
    end

    % must set "work.task = 1"
    workls.task =1;
    deriv = d'*g;
    normd = norm(d);

    stp = 1;
    %       call line search, reverse communication
    while 1

        [stp, f, deriv, parsls, workls] = ls_csrch(stp, f, deriv , parsls , workls);

        % Evaluate the function and the gradient at stp
        if (workls.task == 2)
            % calculate g, f,
            x = xp + stp*d;
            [f,  g] = feval(fun, x, varargin{:});
            Out.nfe = Out.nfe + 1;
            deriv = g'*d;

        else  % exit
            break
        end
    end

    % s = x - xp = stp*d;  ==> ||s|| = stp*||d||
    nrms = stp*normd;
    % compute stopping
    CritdiffX = nrms/max(nrmxp,1);

    % now, update normG
    nrmG =  norm(g);
    Out.nrmg =  nrmG;
    Out.f = [Out.f; f];

    if (opts.debug >= 1)
        fprintf(fid, '%4d \t %4.3e \t %4.3e \t %4.3e \t %4.3e \t %2d\n', iter, stp, f, CritdiffX, nrmG, workls.task);
    end

    %if (CritdiffX < xtol) && (nrmG < gtol)
%     if ((CritdiffX < xtol) && (nrmG < gtol) ) || abs((fp-f)/max(abs([fp,f,1]))) < 1e-20;
    if ((CritdiffX < xtol) || (nrmG < gtol) ) || abs((fp-f)/max(abs([fp,f,1]))) < 1e-20;

        Out.msg = 'converge';
        Out.iter = iter;
        Out.nge = Out.nfe;
        return;
    end

    nrmx = norm(x);

    %----------------------------------------------------------------------
    % save for L-BFGS
    ygk = g-gp;		s = x-xp;

    %Check to save s and y for L-BFGS.
    if ygk'*ygk>1e-20
        istore = istore + 1;
        pos = mod(istore, m); if pos == 0; pos = m; end;    
        YK(:,pos) = ygk;  SK(:,pos) = s;   rho(pos) = 1/(ygk'*s);
        
        if istore <= m; status = istore; perm = [perm, pos]; 
        else status = m; perm = [perm(2:m), perm(1)]; end
    end

end


Out.msg = 'Exceed max iteration';
Out.iter = iter;


% computer y = H*v where H is L-BFGS matrix
    function y = LBFGS_Hg_Loop(dv)
        q = dv;   alpha = zeros(status,1);
        for di = status:-1:1;
            k = perm(di);
            alpha(di) = (q'*SK(:,k)) * rho(k);
            q = q - alpha(di)*YK(:,k);
        end
        y = q/(rho(pos)* (ygk'*ygk));
        for di = 1:status
            k = perm(di);
            beta = rho(k)* (y'* YK(:,k));
            y = y + SK(:,k)*(alpha(di)-beta);
        end
    end

end



