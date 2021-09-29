clear; clc

% Parameters and utility
beta = 0.95; alpha = 0.36; delta = 0.1; sigma_eps = 0.0005;
rho=0.8; mu=0.0; T = 1000;
du=@(c) c.^(-1); duinv=@(u) u.^(-1);

% Simulate exogenous process
rng(10); Mdl = arima('Constant',mu,'AR',rho,'Variance',sigma_eps);
z_sim = exp(simulate(Mdl,T));

% Find the steady state capital and consumption
kss = @(z) [(1-beta*(1-delta))./(beta*alpha.*z)].^(1/(alpha-1));
css = @(k,z) z.*(k.^alpha)-delta*k;
k_init = kss(z_sim);  c_init = css(k_init,z_sim);

% Create and train neural network
net = feedforwardnet(12);
RHS=du(c_init(2:end)).*(alpha.*z_sim(2:end).*k_init(2:end).^(alpha-1)+1-delta);
[net,tr] = train(net,[k_init(1:end-1),z_sim(1:end-1)]',RHS');

% Solve the model
k_old=k_init; k(1)=kss(mean(z_sim));
error=1e+3;
while error>1e-3
    
    for t=1:T
        ERHS(t,1)=net([k(t);z_sim(t)]);
        c(t,1)=duinv(beta*ERHS(t));
        k(t+1,1)=z_sim(t).*k(t).^alpha+(1-delta)*k(t)-c(t);
    end
    k(end)=[];
    
    % Train network
    RHS=du(c(2:end)).*(alpha.*z_sim(2:end).*k(2:end).^(alpha-1)+1-delta);
    [net,y,e] = train(net,[k(1:end-1),z_sim(1:end-1)]',RHS');

    % Check convergence
    error = max(abs(k_old-k))
    k_old = k;
end



