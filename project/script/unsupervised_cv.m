foldCount=5;  %number of folds

loglik=zeros(totalComponents,foldCount);

Nlearning=size(Xtrain,2);
order = randperm(Nlearning);

opts.plotlik=0;
opts.plotsolution=0;
opts.maxit=100;
opts.minDeterminant=0.0001;

for H=1:totalComponents; % number of mixture components 
    for fold=1:foldCount %K-fold cross validation (K=5)

        training_indices = order([1:ceil((fold - 1) * Nlearning / foldCount),...
            ceil(fold * Nlearning / foldCount) + 1:Nlearning]) % cv training index
        val_indices = setdiff(1:Nlearning,training_indices); % cv validation index

        X_train=Xtrain(:,training_indices);  % cv training data
        Xval=Xtrain(:,val_indices); % cv validation data
        %train model
        [P1,m1,S1,loglik1,phgn1]=GMMem(X_train,H,opts); % fit model
        
        %Predict using the cv trained model
        logl1 = GMMloglik(Xval,P1,m1,S1);
        loglik(H,fold) = sum(logl1);
    end
end
% plot the accuracy curve
figure(2);
plot(mean(loglik,2),'bo-');
xlabel('Number of Mixture Components');ylabel('Test Likelihood')
title('Model Selection (Cross Validation)');
[v,h]=max(mean(loglik,2)); %select the number of mixture components which maximizes the accuracy 

%Now train full model with selected number of mixture components
[P1,m1,S1,loglik1,phgn1]=GMMem(Xtrain,h,opts); % fit 

%Predict using the full trained model
logl1=GMMloglik(Xtest,P1,m1,S1);
fprintf('Test Data Likelihood=%f\n',sum(logl1))

% % Plot the best GMM model
% figure(3);
% clf;
% hold on;
% for i = 1:c
%     idc = label==i;
%     plot(Xtrain(1,Xtrain_labels==i),Xtrain(2,Xtrain_labels==i),['.' color(i)],'MarkerSize',12);
% end
% plot(Xtest(1,:),Xtest(2,:),'kd','MarkerSize',5);
% for i=1:h
%     [E V]=eig(S1(:,:,i));dV=sqrt(diag(V)); 
%     theta=0:0.1:2*pi;
%     p(1,:)= dV(1)*cos(theta); p(2,:)= dV(2)*sin(theta);
%     x = E*p+repmat(m1(:,i),1,length(theta));
%     plot(x(1,:),x(2,:),'r-','linewidth',2)
% end;
% title('training data, test data (in black)');
