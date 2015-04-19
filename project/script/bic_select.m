function [ opts, bic_h, aic_h ] = bic_select( X, components )
%BIC Summary of this function goes here
%   Detailed explanation goes here

opts.plotlik=0;
opts.plotsolution=0;
opts.maxit=100;
opts.minDeterminant=0.0001;

ndim = size(X,1);

loglik=zeros(components,1);
BIC = zeros(components,1);
AIC = zeros(components,1);
numParams = zeros(components,1);

for H=1:components; % number of mixture components
    [P1,m1,S1,loglik1,phgn1]=GMMem(X,H,opts); % fit class1 data
    loglik(H)=loglik1;
    numParams(H) = H * ndim*(ndim+1)/2 + H*ndim + (H-1); % number of parameters in the model
    BIC(H) = -2*loglik(H) + numParams(H)*log(size(X,2)); % BIC for the model
    AIC(H) = -2*loglik(H) + 2*numParams(H);
end

%plot the BIC curve
figure;
plot(1:components, BIC,'bo-');
xlabel('Number of Mixture Components');ylabel('BIC')
title('Model Selection (BIC)');
[v,bic_h]=min(BIC); %select the number of mixture components which minimizes the BIC 

%plot the AIC curve
figure;
plot(1:components, AIC,'bo-');
xlabel('Number of Mixture Components');ylabel('AIC')
title('Model Selection (AIC)');
[v, aic_h]=min(AIC); %select the number of mixture components which minimizes the BIC 

end

