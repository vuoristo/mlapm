rng(123123123);

% Simulate data
theta_true = 3;
n_samples = 100;
z = round(rand(n_samples,1)) + 1; % 1 or 2 with equal probability
x = zeros(n_samples,1);
for i = 1:n_samples
    if z(i)==1
        x(i) = randn; % N(0,1)
    elseif z(i)==2
        x(i) = randn + theta_true;  % N(theta_true,1)
    end
end

n_iter = 20;
th0 = 0;
th = zeros(n_iter,1);
th(1) = th0;
for iter = 2:n_iter
    
    % E-step, compute the responsibilities r2 for component 2
    c1_dens = normpdf(x);
    c2_dens = normpdf(x, th(iter-1));
    r2 = c2_dens ./ (c2_dens + c1_dens);
    
    % M-step, compute the parameter value that maximizes
    % the expectation of the complete-data log-likelihood.
    th(iter) = sum(r2 .* x) / sum(r2);
    % Weighted mean of observations, with weight of observation 
    % n given by the posterior probability that component 2 was
    % responsible for generating observation n.
    
end

disp(num2str(th)); % Should converge to the theta_true.