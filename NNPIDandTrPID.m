clc
clear all
close all
%% Traditional PID %%
%**********************************************
% Input signal
h=0.01;
t=0:h:3.5;
for k=1:length(t)
    r(k)=1;
end
u=r;
%**********************************************
% system model
% Initializing y(k)
y(1)=0;
y(2)=0;
Kp = 0.16;
Ki = 0.16;
Kd = 0.025;
for k=3:length(t)
    y(k) = (0.8*y(k-1) + 2*u(k-1))/(1+1.5*y(k-2)*u(k-2));
     ey(k)=r(k)-y(k);   %error
    eyc(k,1)=ey(k)-ey(k-1);
    u(k)=u(k-1)+Kp*(ey(k)-ey(k-1))+Ki*(ey(k))+Kd*(ey(k)-2*ey(k-1)+ey(k-2));
    loss_PID = 0.5*ey(k)^2;
    %%disturbance
     if t(k-1)==1
        u(k)=u(k)+0.5; %first dis
    elseif t(k-1)==1.8
        u(k)=u(k)+0.89; %second dis
    end
    
end



%% NNPID %%

%% Input signal
h=0.002;
t_np=0:h:3.5;
t_np=t_np';
for k_np=1:length(t_np)
    r_np(k_np,1)=1;
    rp_np(k_np,1)=4;
    
end
Kp_np=0.3;
Ki_np=0.2;
Kd_np=0.01;
B=1; 
%% neural network initialization
N=20; % number of hidden neurons
dim=4;  %number of inputs
eta=0.01; % learning rate
alfa=0.04; % momentom
Wij=0.05*rand(N,dim);  %input to hidden weights
Wli=0.05*rand(3,N);   %hidden to output weights
dWli(:,:,2)=zeros(3,N);
dWij(:,:,2)=zeros(N,dim);

%%
% model
% Initializing y(k) & u(k) & e(k) for k=1,2
y_np(1,1)=0;
y_np(2,1)=0;

e(1,1)=r_np(1)-y_np(1);
e(2,1)=r_np(2)-y_np(2);

u_np(1,1)=0;
u_np(2,1)=0;


for k_np=3:length(t_np)

% % plant & PID & error
    y_np(k_np,1) = (0.8*y_np(k_np-1) + 2*u_np(k_np-1))/(1+1.5*y_np(k_np-2)*u_np(k_np-2));
    e(k_np)=r_np(k_np)-y_np(k_np);   %error
    e2(k_np,1)=e(k_np)-e(k_np-1);
    u_np(k_np)=u_np(k_np-1)+Kp_np*(e(k_np)-e(k_np-1))+Ki_np*(e(k_np))+Kd_np*(e(k_np)-2*e(k_np-1)+e(k_np-2)); %PID 
    loss(k_np) = 0.5*e(k_np)^2;
%     loss = mean(loss);
%% disturbance
    if t_np(k_np-1)==1
        u_np(k_np,1)=u_np(k_np,1)+0.5; %first dis
    elseif t_np(k_np-1)==1.8
        u_np(k_np,1)=u_np(k_np,1)+0.89; %second dis
    end
    %% neural network
    % input layer
    x=[r_np(k_np);u_np(k_np);e(k_np);e2(k_np,1)];  
    O1=x;
    
    %hidden layyer
    net2=Wij*O1;
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
                Delta3(l)=e(k_np)*sign(gradient(y_np(k_np),u_np(k_np)))*(e(k_np)-e(k_np-1))*gradactivLJ;
            elseif l==2
                Delta3(l)=e(k_np)*sign(gradient(y_np(k_np),u_np(k_np)))*(e(k_np))*gradactivLJ;
            elseif l==3
                Delta3(l)=e(k_np)*sign(gradient(y_np(k_np),u_np(k_np)))*(e(k_np)-2*e(k_np-1)+e(k_np-2))*gradactivLJ;
            end
            
            dWli(l,i,k_np)=alfa* dWli(l,i,k_np-1)+eta*Delta3(l)*O2(i);
            Wli_new(l,i)=Wli(l,i)+dWli(l,i,k_np);
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
            
            dWij(i,j,k_np)=alfa*dWij(i,j,k_np-1)+eta*Delta2(i)*O1(j);
            Wij(i,j)=Wij(i,j)+dWij(i,j,k_np);
        end
    end
    
    Wli=Wli_new;
                
    net2=Wij*O1;
    O2=tanh(net2);
    
    net3=Wli*O2;
    O3new=0.5*(1+tanh(net3))./[3.33;3.33;20];
    K_np(:,k_np)=O3new;
%     O3 = O3new;

    Kp_np=O3new(1);
    Ki_np=O3new(2);
    Kd_np=O3new(3);
    K_np(1,k_np)=Kp_np;
    K_np(2,k_np)=Ki_np;
    K_np(3,k_np)=Kd_np;
    k_np = k_np+1;
end



%% plotting
%%
figure('Name','System Output with Traditional PID','NumberTitle','off');
plot(t,y,'b','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)');
ylabel('y_{PID}(k)');
title('System Output with Traditional PID');
legend('y_{PID}(k)')
%%
figure('Name','Control Signal with Traditional PID','NumberTitle','off');
plot(t,u,'g','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('u_{PID}(k)')
title('Control Signal With Traditional PID')
legend('u_{PID}(k)')
%%
figure('Name','System Output with NN-PID','NumberTitle','off');
plot(t_np,y_np,'r','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('y_{NN-PID}(k)')
title('System Output with NN-PID')
legend('y_{NN-PID}(k)')
%%
figure('Name','Control Signal with NN-PID','NumberTitle','off');
plot(t_np,u_np,'m','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('u_{NN-PID}(k)')
title('Control Signal with NN-PID')
legend('u_{NN-PID}(k)')
%%
figure('Name','Loss Function with NN-PID','NumberTitle','off');
plot(t_np,loss,'k','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('Loss_{NN-PID}')
title('Loss function')
legend('Loss_{NN-PID}')
%%
figure('Name','PID Gains with NN-PID','NumberTitle','off');
subplot(3,1,1)
plot(t_np,K_np(1,:),'r','LineWidth',2)
grid on
set(gca,'FontSize',15)
axis([-0.05 max(t) 0.05 0.2])
xlabel('time(s)')
ylabel('K_p');
title('PID Gains with NN-PID')

subplot(3,1,2)
plot(t_np,K_np(2,:),'g','LineWidth',2)
grid on
set(gca,'FontSize',15)
axis([-0.05 max(t) 0.05 0.2])
xlabel('time(s)')
ylabel('K_i')

subplot(3,1,3)
plot(t_np,K_np(3,:),'b','LineWidth',2)
grid on
set(gca,'yTick',0.005:0.005:0.025,'FontSize',15)
axis([-0.05 max(t) 0.005 0.03])
xlabel('time(s)')
ylabel('K_d')
%%
figure('Name','System Output','NumberTitle','off');
plot(t,y,'LineWidth',2);
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)');
ylabel('y(k)');
title('System Output');
hold on
plot(t_np,y_np,'LineWidth',2);
hold off
legend('y_{PID}(k)','y_{NN-PID}(k)')
%%
figure('Name','System Output','NumberTitle','off');
subplot(2,1,1)
plot(t,y,'b','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)');
ylabel('y_{PID}(k)');
title('System Output with Traditional PID');
legend('y_{PID}(k)')
subplot(2,1,2)
plot(t_np,y_np,'r','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('y_{NN-PID}(k)')
title('System Output with NN-PID')
legend('y_{NN-PID}(k)')
%%
figure('Name','Control Signal','NumberTitle','off');
plot(t,u,'LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('u(k)')
title('Control Signal')
hold on
plot(t_np,u_np,'LineWidth',2)
hold off
legend('u_{PID}(k)','u_{NN-PID}(k)')
%%
figure('Name','Control Signal','NumberTitle','off');
subplot(2,1,1)
plot(t,u,'g','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('u_{PID}(k)')
title('Control Signal With Traditional PID')
legend('u_{PID}(k)')
subplot(2,1,2)
plot(t_np,u_np,'m','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('u_{NN-PID}(k)')
title('Control Signal with NN-PID')
legend('u_{NN-PID}(k)')


%%
figure('Name','loss','NumberTitle','off');
subplot(2,1,1)
plot(t,loss_PID,'g','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('loss_{PID}(k)')
title('loss Traditional PID')
legend('loss_{PID}(k)')
subplot(2,1,2)
plot(t_np,loss,'m','LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('loss_{NN-PID}(k)')
title('loss of NN-PID')
legend('loss_{NN-PID}(k)')
%%
figure('Name','loss','NumberTitle','off');
plot(t,loss_PID,'LineWidth',2)
grid on
set(gca,'FontSize',15)
xlim([-0.01 max(t)])
xlabel('time(s)')
ylabel('loss_PID')
title('loss')
hold on
plot(t_np,loss,'LineWidth',2)
hold off
legend('loss_{PID}(k)','loss_{NN-PID}(k)')

