clear all;
close all;
load projectdata;

opts.plotlik=0;
opts.plotsolution=0;
opts.maxit=100;
opts.minDeterminant=0.0001;

class1Data = trainData(trainLabels==1,:);
class2Data = trainData(trainLabels==2,:);
class3Data = trainData(trainLabels==3:124,:);
class5Data = trainData(trainLabels==5,:);
class6Data = trainData(trainLabels==6,:);
class7Data = trainData(trainLabels==7,:);

[P1,m1,S1,loglik1,phgn1]=GMMem(class1Data,1,opts); % fit class1 data
[P2,m2,S2,loglik2,phgn2]=GMMem(class2Data,1,opts); % fit class2 data
[P3,m3,S3,loglik3,phgn3]=GMMem(class3Data,1,opts); % fit class3 data
[P5,m5,S5,loglik5,phgn5]=GMMem(class5Data,1,opts); % fit class5 data
[P6,m6,S6,loglik6,phgn6]=GMMem(class6Data,1,opts); % fit class6 data
[P7,m7,S7,loglik7,phgn7]=GMMem(class7Data,1,opts); % fit class7 data

for i = 1:size(testData,1)
    xtest = testData(i,:);
    logl1=GMMloglik(xtest,P1,m1,S1);
    logl2=GMMloglik(xtest,P2,m2,S2);
    logl3=GMMloglik(xtest,P3,m3,S3);
    logl5=GMMloglik(xtest,P5,m5,S5);
    logl6=GMMloglik(xtest,P6,m6,S6);
    logl7=GMMloglik(xtest,P7,m7,S7);
    condexp([logl1 ; logl2 ; logl3 ; logl5 ; logl6 ; logl7])
end

for i=1:5
    n = 5*i;
    knn_mdl = fitcknn(trainData,trainLabels, 'NumNeighbors', n);
    predicted_labels = knn_mdl.predict(testData);

    correct = predicted_labels == testLabels;
    accuracy(i) = sum(correct)/length(testLabels);
end