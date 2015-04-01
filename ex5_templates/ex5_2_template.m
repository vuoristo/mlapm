clear all;
rand('state', 999); %#ok<RAND>
randn('state', 999); %#ok<RAND>
% Make some simulated training data X :
l = 0.2; r1 = 0.5;
for r = 1:100
	rad = r1 + rand*l;  theta = rand*2*pi;
	X(1,r) = rad*cos(theta); X(2,r) = rad*sin(theta);
end

% Make test data point :
xtest=[-0.4 0.2]';

%Visualize training data
plot(X(1,:),X(2,:),'o'); hold on;
figure(1); plot(xtest(1,1),xtest(2,1),'kd');
title('training data(in blue), test data(in black)');

% here you can use ex5_1_template for traing the GMM model
% you will get the S, m and P paratemer values from the template
% given the test point xtest (above as test data point)
% and following % prior probabilities 
% infer the posterior probabilities for the test point for each of the
% mixture components

prProb = size(xtest,1); % prior probabilities
for i = 1:H
% here you need to write your own code for computing the posterior
% probabilities for the test point
end

%postProb  = ?; % posterior probabilities
figure; bar(postProb); title('Posterior probabilities for the test point')
