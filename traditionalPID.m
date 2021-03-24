%% 
%Farshid A'azam Manesh 97442052


%% Open loop responce of IAQ
clc
clear all
close all
%**********************************************
% Input signal
h=0.01;
t=0:h:3.5;
for k=1:length(t)
    r(k)=1;
end
u=r;
%**********************************************
% IAQ model
% Initializing y(k)
y(1)=0;
y(2)=0;
Kp = 0.16;
Ki = 0.16;
Kd = 0.025;
for k=3:length(t)
    y(k,1) = (0.8*y(k-1) + 2*u(k-1))/(1+1.5*y(k-2)*u(k-2));
     ey(k)=r(k)-y(k);   %error
    eyc(k,1)=ey(k)-ey(k-1);
    u(k)=u(k-1)+Kp*(ey(k)-ey(k-1))+Ki*(ey(k))+Kd*(ey(k)-2*ey(k-1)+ey(k-2));
    
     if t(k-1)==1
        u(k)=u(k)+0.5; %first dis
    elseif t(k-1)==1.8
        u(k)=u(k)+0.89; %second dis
    end
    
end

plot(t,y,'g','LineWidth',1.5)
axis([-0.01 3.5 0 2])
xlabel('time(s)');
ylabel('y(k)');
title('system ouput with traditional PID');
legend('y(k)')












