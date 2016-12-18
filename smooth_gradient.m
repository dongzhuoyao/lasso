function [result] = smooth_gradient(x)
t = 1e-4;
if abs(x)<=t
    result = x/t;
else 
    result = sign(x);
end