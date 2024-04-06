clc;clear;close all
% load dataset
[u,y] = maglev_dataset;

d1 = [1:2];
d2 = [1:2];
narx_net = narxnet(d1,d2,10);
narx_net.divideFcn = '';
narx_net.trainParam.min_grad = 1e-10;
[p,Pi,Ai,t] = preparets(narx_net,u,{},y);

narx_net = train(narx_net,p,t,Pi);

return