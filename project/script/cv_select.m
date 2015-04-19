function loglik = cv_select( X, components, fold_count )
%CV_SELECT Summary of this function goes here
%   Detailed explanation goes here

opts.plotlik=0;
opts.plotsolution=0;
opts.maxit=200;
opts.minDeterminant=0.0001;

loglik=zeros(components,fold_count);

Nlearning=size(X,2);
order = randperm(Nlearning);

for H=1:components; % number of mixture components 
    for fold=1:fold_count %K-fold cross validation (K=5)

        training_indices = order([1:ceil((fold - 1) * Nlearning / fold_count),...
            ceil(fold * Nlearning / fold_count) + 1:Nlearning]); % cv training index
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

% [v,h]=max(mean(loglik,2)); %select the number of mixture components which maximizes the accuracy 

end

