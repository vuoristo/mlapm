ndim = size(testData,2);

loglik=zeros(totalComponents,1);
BIC = zeros(totalComponents,1);
numParams = zeros(totalComponents,1);

opts.plotlik=0;
opts.plotsolution=0;
opts.maxit=100;
opts.minDeterminant=0.0001;

for H=1:totalComponents; % number of mixture components
        [P1,m1,S1,loglik1,phgn1]=GMMem(Xtrain,H,opts); % fit class1 data
        loglik(H)=loglik1;
        numParams(H) = H * ndim*(ndim+1)/2 + H*ndim + (H-1); % number of parameters in the model
        BIC(H) = -2*loglik(H) + numParams(H)*log(size(Xtrain,2)); % BIC for the model
end

%plot the BIC curve
figure(2);
plot(1:totalComponents, BIC,'bo-');
xlabel('Number of Mixture Components');ylabel('BIC')
title('Model Selection (BIC)');
[v,h]=min(BIC); %select the number of mixture components which minimizes the BIC 

%Now train full model with selected number of mixture components
[P1,m1,S1,loglik1,phgn1]=GMMem(Xtrain,h,opts); % fit to data

%Predict using the full trained model
logl1=GMMloglik(Xtest,P1,m1,S1);
fprintf('Test Data Likelihood=%f\n',sum(logl1))
