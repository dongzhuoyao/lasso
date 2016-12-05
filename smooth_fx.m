function [result] = smooth_fx(x)
t = 0.01;
if abs(x)<=t
    result = x^2/t/2.0;
else 
    result = sign(x)-t/2.0;
end