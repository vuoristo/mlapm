clear all;
close all;

rand('state', 999); %#ok<RAND>
randn('state', 999); %#ok<RAND>

load projectdata;

complete_data = [trainData; testData];
complete_labels = [trainLabels; testLabels];

totalComponents = 10; %max number of mixture components

[bic_opts, bic_h, aic_h] = bic_select(complete_data', totalComponents);
[cv_opts, cv_h] = cv_select(complete_data', totalComponents);

% %Now train full model with selected number of mixture components
% [P1,m1,S1,loglik1,phgn1]=GMMem(data,h,opts); % fit to data
% 
% %Predict using the full trained model
% logl1=GMMloglik(Xtest,P1,m1,S1);
% fprintf('Test Data Likelihood=%f\n',sum(logl1))