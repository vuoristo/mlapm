clear all;
close all;

MAX_COMPONENTS = 10; %max number of mixture components
FOLD_COUNT = 5;
COLORS = 'rbgy';
PCA_N = [2, 4, 5]

rand('state', 999); %#ok<RAND>
randn('state', 999); %#ok<RAND>

load projectdata;

complete_data = [trainData; testData];
complete_labels = [trainLabels; testLabels];

figure(1);
title('Model Selection (BIC)');
xlabel('Number of Mixture Components');ylabel('BIC');
legend('1','2','4','5');

figure(2);
title('Model Selection (AIC)');
legend('1','2','4','5');
xlabel('Number of Mixture Components');ylabel('AIC');

figure(3);
title('Model Selection (Cross Validation)');
xlabel('Number of Mixture Components');ylabel('Likelihood');
legend('1','2','4','5');

% Model selection using the full data
[bic, aic] = bic_select(complete_data', MAX_COMPONENTS);
figure(1); hold on;
plot(1:MAX_COMPONENTS, bic, [COLORS(1), '-o']);
figure(2); hold on;
plot(1:MAX_COMPONENTS, aic, [COLORS(1), '-o']);

loglik = cv_select(complete_data', MAX_COMPONENTS, FOLD_COUNT);
figure(3); hold on;
plot(1:MAX_COMPONENTS, mean(loglik, 2), [COLORS(1), '-o']);

% Model selection using dimension reduced data
for i=1:length(PCA_N)
    pca_n = PCA_N(i);
    [coeff, pca_data, latent, tsquared, explained, mu] = ...
        pca(complete_data, 'NumComponents', pca_n);

    [bic, aic] = bic_select(pca_data', MAX_COMPONENTS);
    figure(1); hold on;
    plot(1:MAX_COMPONENTS, bic, [COLORS(i+1), '-o']);
    figure(2); hold on;
    plot(1:MAX_COMPONENTS, aic, [COLORS(i+1), '-o']);

    loglik = cv_select(pca_data', MAX_COMPONENTS, FOLD_COUNT);
    figure(3); hold on;
    plot(1:MAX_COMPONENTS, mean(loglik, 2), [COLORS(i+1), '-o']);
end


% %Now train full model with selected number of mixture components
% [P1,m1,S1,loglik1,phgn1]=GMMem(data,h,opts); % fit to data
% 
% %Predict using the full trained model
% logl1=GMMloglik(Xtest,P1,m1,S1);
% fprintf('Test Data Likelihood=%f\n',sum(logl1))