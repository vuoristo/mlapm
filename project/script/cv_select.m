function [ opts, h ] = cv_select( X, components )
%CV_SELECT Summary of this function goes here
%   Detailed explanation goes here

FOLD_COUNT=5;  %number of folds

opts.plotlik=0;
opts.plotsolution=0;
opts.maxit=200;
opts.minDeterminant=0.0001;

loglik=zeros(components,FOLD_COUNT);

Nlearning=size(X,2);
order = randperm(Nlearning);

for H=1:components; % number of mixture components 
    for fold=1:FOLD_COUNT %K-fold cross validation (K=5)

        training_indices = order([1:ceil((fold - 1) * Nlearning / FOLD_COUNT),...
            ceil(fold * Nlearning / FOLD_COUNT) + 1:Nlearning]); % cv training index
        val_indices = setdiff(1:Nlearning,training_indices); % cv validation index

        X_train=X(:,training_indices);  % cv training data
        Xval=X(:,val_indices); % cv validation data
        %train model
        [P1,m1,S1,loglik1,phgn1]=GMMem(X_train,H,opts); % fit model
        
        %Predict using the cv trained model
        logl1 = GMMloglik(Xval,P1,m1,S1);
        loglik(H,fold) = sum(logl1);
    end
end

% plot the accuracy curve
figure;
plot(mean(loglik,2),'bo-');
xlabel('Number of Mixture Components');ylabel('Test Likelihood');
title('Model Selection (Cross Validation)');
[v,h]=max(mean(loglik,2)); %select the number of mixture components which maximizes the accuracy 

end

