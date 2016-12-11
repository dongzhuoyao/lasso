function [result] = shrinkage(u,t)
if u>t
    result = u-t;
else if -t<= u && u<=t
    result=0;
else
    result=u+t;
    end

end