function [result] = my_sign(x)
if x>0.002
    result=1;
else if x<-0.002
    result=-1;
else
    result=0;
    end
end