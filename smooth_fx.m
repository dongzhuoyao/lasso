function [result] = smooth_fx(x)
t = 1e-4;
if abs(x)<=t
    result = x^2/t/2.0;
else 
    result = abs(x)-t/2.0;
end