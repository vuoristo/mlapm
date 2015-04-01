clear all;
rand('state', 999); %#ok<RAND>
randn('state', 999); %#ok<RAND>
%clf;
% Make some simulated data :
% l = 0.2; r1 = 0.5;
% for r = 1:100
% 	rad = r1 + rand*l;  theta = rand*2*pi;
% 	X(1,r) = rad*cos(theta); X(2,r) = rad*sin(theta);
% end

%Visualize simulated data
% figure(1);
% plot(X(1,:),X(2,:),'o');
% title('Data');
load('projectdata');
X = testData;
D = size(X,1); % dimension of the space
N = size(X,2); % number of data points

%Initialize parameters
H = 7; %number of mixture components
r = randperm(N); 
m = X(:,r(1:H)); % initialise the centres to random datapoints
s2 = mean(diag(cov(X')));
S = repmat(s2*eye(D),[1 1 H]); % initialise the variances to be large
P = ones(H,1)./H;  % intialise the component probilities to be uniform
minDeterminant=0.0001;
%Initialize number of iterations needed for convergence
numOfIteration=50;


for iter = 1:numOfIteration
	% E-step:
	% Here you need to write your code for the E-step of the EM Algorithm 
	for i = 1:H
		invSi = inv(S(:,:,i));
		[u s v]=svd(2*pi*S(:,:,i)); 
        logdetSi=sum(log(diag(s)+1.0e-20));
		for n = 1:N
			v = X(:,n) - m(:,i);
			logpold(n,i) =-0.5*v'*invSi*v - 0.5*logdetSi + log(P(i));
		end
    end
    phgn=condexp(logpold'); % responsibilities
	pngh = condp(phgn'); % membership

	logl(iter)=0;
	for n=1:N
        % Compute the log likelihood for current iteration 
		logl(iter)= logl(iter) + logsumexp(logpold(n,:),ones(1,H));
    end
 	% M-step:
    % M-step:
    % Here you need to write your code for the M-step of the EM Algorithm 
    % and update the following parameters
    %S = ?; %learned Covariances%
    %m = ?; %learned Means
	%P = ?; %learned Mixture Coefficients
	for i = 1:H % now get the new parameters for each component
		tmp = (X - repmat(m(:,i),1,N)).*repmat(sqrt(pngh(:,i)'),D,1);
		Scand =  tmp*tmp';
		if det(Scand)> minDeterminant % don't accept too low determinant
			S(:,:,i) = Scand; %learned Covariances
		end
    end
    m = X*pngh; %learned Means
	P = sum(phgn,2)/N; %learned Mixture Coefficients
end
%Visualize log likelihood over the iterations
figure(2);
plot(logl,'-o'); title('log likelihood');

%Visualize the trained mixture model
% figure(3);
% plot(X(1,:),X(2,:),'o'); hold on;
% for i=1:H
%     [E V]=eig(S(:,:,i));dV=sqrt(diag(V)); 
%     theta=0:0.3:2*pi;
%     p(1,:)= dV(1)*cos(theta); p(2,:)= dV(2)*sin(theta);
%     x = E(1:2,1:2)*p+repmat(m(1:2,i),1,length(theta));
%     plot(x(1,:),x(2,:),'r-','linewidth',2)
% end;
% title('Trained GMM Model');
% drawnow;
hold off