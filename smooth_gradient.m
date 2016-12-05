function [result] = smooth_gradient(x)
t = 0.01;
if abs(x)<=t
    result = x/t;
else 
    result = sign(x);
end