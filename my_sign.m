function [result] = my_sign(x,t)
if x>t
    result=1;
else if x<-t
    result=-1;
else
    result=0;
    end
end