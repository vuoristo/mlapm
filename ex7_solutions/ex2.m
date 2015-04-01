% Exercise problem 2: Laplace and BIC approximations

%% First with n=5
n = 5;
x1 = 2; x2 = 1;

% M1 MAP estimate
p = (x1+x2)/(2*n);
% M2 MAP estimate
p1 = x1/n; p2 = x2/n;

% hessian M1
H = (x1+x2)/p^2 + (2*n-x1-x2)/(1-p)^2;
% hessian M2
H1 = x1/p1^2+(n-x1)/(1-p1)^2;
H2 = x2/p2^2+(n-x2)/(1-p2)^2;

% Laplace approximations
L1 = log(binopdf(x1,n,p)) + log(binopdf(x2,n,p)) + log(betapdf(p,1,1)) ...
    + .5*log(2*pi) -.5*log(H);
L2 = log(binopdf(x1,n,p1)) + log(binopdf(x2,n,p2)) + log(betapdf(p1,1,1)) ...
    + log(betapdf(p2,1,1)) + 1*log(2*pi) -.5*log(H1) -.5*log(H2);
BF_laplace = exp(L1-L2)

% BIC
bic1 = log(binopdf(x1,n,p)) + log(binopdf(x2,n,p)) - 1/2*log(2*n);
bic2 = log(binopdf(x1,n,p1)) + log(binopdf(x2,n,p2)) - 2/2*log(2*n);
BF_bic = exp(bic1-bic2)

%% Then with n=500
n = 500;
x1 = 200; x2 = 250;

% M1 MAP estimate
p = (x1+x2)/(2*n);
% M2 MAP estimate
p1 = x1/n; p2 = x2/n;

% hessian M1
H = (x1+x2)/p^2 + (2*n-x1-x2)/(1-p)^2;
% hessian M2
H1 = x1/p1^2+(n-x1)/(1-p1)^2;
H2 = x2/p2^2+(n-x2)/(1-p2)^2;

% Laplace approximations
L1 = log(binopdf(x1,n,p)) + log(binopdf(x2,n,p)) + log(betapdf(p,1,1)) ...
    + .5*log(2*pi) -.5*log(H);
L2 = log(binopdf(x1,n,p1)) + log(binopdf(x2,n,p2)) + log(betapdf(p1,1,1)) ...
    + log(betapdf(p2,1,1)) + 1*log(2*pi) -.5*log(H1) -.5*log(H2);
BF_laplace = exp(L1-L2)

% BIC
bic1 = log(binopdf(x1,n,p)) + log(binopdf(x2,n,p)) - 1/2*log(2*n);
bic2 = log(binopdf(x1,n,p1)) + log(binopdf(x2,n,p2)) - 2/2*log(2*n);
BF_bic = exp(bic1-bic2)