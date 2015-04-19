clear all;
close all;

MAX_COMPONENTS = 10; %max number of mixture components
FOLD_COUNT = 5;
COLORS = 'rbgy';
PCA_N = [2, 4, 5];

rand('state', 999); %#ok<RAND>
randn('state', 999); %#ok<RAND>

load projectdata;

complete_data = [trainData; testData];
complete_labels = [trainLabels; testLabels];

figure(1);
title('Model Selection (BIC)');
xlabel('Number of Mixture Components');ylabel('BIC');

figure(2);
title('Model Selection (AIC)');
xlabel('Number of Mixture Components');ylabel('AIC');

figure(3);
title('Model Selection (Cross Validation)');
xlabel('Number of Mixture Components');ylabel('Likelihood');

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
% H's is the suggested count of GMM components for each
% criterion/dimension combo
HS = zeros(length(PCA_N+1), 3);
for i=1:length(PCA_N)
    pca_n = PCA_N(i);
    [coeff, pca_data, latent, tsquared, explained, mu] = ...
        pca(complete_data, 'NumComponents', pca_n);

    [bic, aic] = bic_select(pca_data', MAX_COMPONENTS);
    [v, bic_h]=min(bic);
    HS(i, 1) = bic_h;
    [v, aic_h]=min(aic);
    HS(i, 2) = aic_h;

    figure(1); hold on;
    plot(1:MAX_COMPONENTS, bic, [COLORS(i+1), '-o']);
    figure(2); hold on;
    plot(1:MAX_COMPONENTS, aic, [COLORS(i+1), '-o']);

    loglik = cv_select(pca_data', MAX_COMPONENTS, FOLD_COUNT);
    [v,cv_h]=max(mean(loglik,2));
    HS(i, 3) = cv_h;

    figure(3); hold on;
    plot(1:MAX_COMPONENTS, mean(loglik, 2), [COLORS(i+1), '-o']);
end

figure(1); legend('1','2','4','5');
figure(2); legend('1','2','4','5');
figure(3); legend('1','2','4','5');

opts.plotlik=0;
opts.plotsolution=0;
opts.maxit=200;
opts.minDeterminant=0.0001;

% Fit GMM's with different amount of mixture components
% and compare the predictions against the ground-truth
% labels.
for h=1:10
    [P1,m1,S1,loglik1,phgn1]=GMMem(complete_data',h,opts); % fit to data
    [Y,I] = max(phgn1); % I is the point class list

    correct = (I == complete_labels);
    accuracy(h) = sum(correct) / length(complete_labels);

%     %Predict using the full trained model
%     logl1=GMMloglik(testData,P1,m1,S1);
%     fprintf('Test Data Likelihood=%f\n',sum(logl1))
end
