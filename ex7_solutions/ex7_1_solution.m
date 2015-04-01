rand('state', 999); %#ok<RAND>
randn('state', 999); %#ok<RAND>

totalComponents = 10; %max number of mixture components
load projectdata;
ratio=0.75;

% train_Index = randperm(size(X,2),ceil(ratio*size(X,2))); %training data index
% test_indices= setdiff(1:size(X,2),train_Index); %test data index

% Xtrain=X(:,train_Index); %training data
% Xtrain_labels=label(train_Index); %training data labels

Xtrain = trainData';
Xtrain_labels = trainLabels;

% Xtest=X(:,test_indices); %test data
% Xtest_labels=label(test_indices); %test data labels

Xtest = testData';
Xtest_labels = testLabels;

ndim = size(testData,2);

%plot training and test data
% color = 'brgmcyk';
% m = length(color);
% c = max(Xtrain_labels);
% figure(1);
% clf;
% hold on;
% for i = 1:c
%     idc = label==i;
%     plot(Xtrain(1,Xtrain_labels==i),Xtrain(2,Xtrain_labels==i),['.' color(i)],'MarkerSize',12);
% end
% plot(Xtest(1,:),Xtest(2,:),'kd','MarkerSize',5);
% title('training data, test data (in black)');

loglik=zeros(totalComponents,1);
BIC = zeros(totalComponents,1);
numParams = zeros(totalComponents,1);

for H=1:totalComponents; % number of mixture components
        opts.plotlik=0;
        opts.plotsolution=0;
        opts.maxit=100;
        opts.minDeterminant=0.0001;
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

% Plot the best GMM model
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
