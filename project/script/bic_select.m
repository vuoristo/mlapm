function [ opts, h ] = bic_select( data, components )
%BIC Summary of this function goes here
%   Detailed explanation goes here

opts.plotlik=0;
opts.plotsolution=0;
opts.maxit=100;
opts.minDeterminant=0.0001;

ndim = size(data,2);

loglik=zeros(components,1);
BIC = zeros(components,1);
numParams = zeros(components,1);

for H=1:components; % number of mixture components
        [P1,m1,S1,loglik1,phgn1]=GMMem(data,H,opts); % fit class1 data
        loglik(H)=loglik1;
        numParams(H) = H * ndim*(ndim+1)/2 + H*ndim + (H-1); % number of parameters in the model
        BIC(H) = -2*loglik(H) + numParams(H)*log(size(data,2)); % BIC for the model
end

%plot the BIC curve
figure;
plot(1:components, BIC,'bo-');
xlabel('Number of Mixture Components');ylabel('BIC')
title('Model Selection (BIC)');
[v,h]=min(BIC); %select the number of mixture components which minimizes the BIC 

end

