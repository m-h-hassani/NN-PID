clc
clear all
close all
%% Input signal
h=0.002;
t=0:h:3.5;
t=t';
for k=1:length(t)
    r(k,1)=1;
    rp(k,1)=4;
    
end
Kp=0.3;
Ki=0.2;
Kd=0.01;
B=1; 
%% neural network initialization
N=20; % number of hidden neuron
dim=4;
eta=0.01; % learning rate
alfa=0.04; % momentom
Wij=0.05*rand(N,dim);  %input to hidden weights
Wli=0.05*rand(3,N);   %hidden to output weights
dWli(:,:,2)=zeros(3,N);
dWij(:,:,2)=zeros(N,dim);

%%
% model
% Initializing y(k) & u(k) & e(k) for k=1,2
y(1,1)=0;
y(2,1)=0;

e(1,1)=r(1)-y(1);
e(2,1)=r(2)-y(2);

u(1,1)=0;
u(2,1)=0;


for k=3:length(t)

% % plant & PID & error
    y(k,1) = (0.8*y(k-1) + 2*u(k-1))/(1+1.5*y(k-2)*u(k-2));
    e(k)=r(k)-y(k);   %error
    e2(k,1)=e(k)-e(k-1);
    u(k)=u(k-1)+Kp*(e(k)-e(k-1))+Ki*(e(k))+Kd*(e(k)-2*e(k-1)+e(k-2)); %PID 
    loss(k) = 0.5*e(k)^2;
%     loss = mean(loss);
%% disturbance
    if t(k-1)==1
        u(k,1)=u(k,1)+0.5; %first dis
    elseif t(k-1)==1.8
        u(k,1)=u(k,1)+0.89; %second dis
    end
    %% neural network
    % input layer
    x=[r(k);u(k);e(k);e2(k,1)];  
    O1=x;
    
    %hidden layyer
    net2=Wij*O1;
%     O2=tanh(net2);
    O2 =net2/(1 + exp(B*net2));
    
    %outout layer
    net3=Wli*O2;
    O3=0.5*(1+tanh(net3));
    
    
    %% updating weights between hidden and output layer
    for i=1:N
        for l=1:3
            activLJ=0.5+0.5*tanh(net3(l));
            gradactivLJ=activLJ*(1-activLJ);
%        Sig = 1/(1+exp(B*net3(l)));
%        activLJ = net3(l)/(1 + exp(B*net3(l)));    % SWISH activation function
%        gradactivLJ = B*activLJ + Sig*(1-B*activLJ);  %gradient of activation function
            if l==1
                Delta3(l)=e(k)*sign(gradient(y(k),u(k)))*(e(k)-e(k-1))*gradactivLJ;
            elseif l==2
                Delta3(l)=e(k)*sign(gradient(y(k),u(k)))*(e(k))*gradactivLJ;
            elseif l==3
                Delta3(l)=e(k)*sign(gradient(y(k),u(k)))*(e(k)-2*e(k-1)+e(k-2))*gradactivLJ;
            end
            
            dWli(l,i,k)=alfa* dWli(l,i,k-1)+eta*Delta3(l)*O2(i);
            Wli_new(l,i)=Wli(l,i)+dWli(l,i,k);
        end
    end
    %% updating weights between input and hidden layer
    for i=1:N
        for j=1:dim
%             f=tanh(net2(i)); 
%             fp=0.5-0.5*f*f;
             Sig = 1/(1+exp(B*net2(i)));
             activIJ = net2(i)/(1 + exp(B*net2(i)));    % SWISH activation function
             gradactivIJ = B*activIJ + Sig*(1-B*activIJ);  %gradient of activation function
            for l=1:3
                S(l)=Delta3(l)*Wli(l,i);
            end
            Delta2(i)=gradactivIJ*sum(S);
            
            dWij(i,j,k)=alfa*dWij(i,j,k-1)+eta*Delta2(i)*O1(j);
            Wij(i,j)=Wij(i,j)+dWij(i,j,k);
        end
    end
    
    Wli=Wli_new;
                
    net2=Wij*O1;
    O2=tanh(net2);
    
    net3=Wli*O2;
    O3new=0.5*(1+tanh(net3))./[3.33;3.33;20];
    K(:,k)=O3new;
%     O3 = O3new;

    Kp=O3new(1);
    Ki=O3new(2);
    Kd=O3new(3);
    K(1,k)=Kp;
    K(2,k)=Ki;
    K(3,k)=Kd;
    k = k+1;
end

%% plot

plot(t,y,'r','LineWidth',1.5)
axis([-0.05 2.5 -0 2.4])
xlabel('time(s)')
ylabel('y(k)')
title('system output with NN-PID')
legend('y(k)')

figure
plot(t,u,'b','LineWidth',1.5)
axis([-0.05 2.5 -0 1])
xlabel('time(s)')
ylabel('u(k)')
title('control signal')
legend('u(k)')

figure
plot(t,loss,'b','LineWidth',1.5)
axis([-0.05 2.5 -0 0.5])
xlabel('time(s)')
ylabel('loss')
title('loss function')
legend('loss')


figure
subplot(3,1,1)
plot(t,K(1,:),'r','LineWidth',1.5)
axis([-0.05 2.5 0.05 0.2])
xlabel('time(s)')
ylabel('Kp');
set(gca,'yTick',0.1:0.05:0.2)

subplot(3,1,2)
plot(t,K(2,:),'y','LineWidth',1.5)
axis([-0.05 2.5 0.05 0.2])
xlabel('time(s)')
ylabel('Ki')
set(gca,'yTick',0.05:0.05:0.15)

subplot(3,1,3)
plot(t,K(3,:),'g','LineWidth',1.5)
axis([-0.05 2.5 0.005 0.03])
xlabel('time(s)')
ylabel('Kd')
set(gca,'yTick',0.005:0.005:0.025)
