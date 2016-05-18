
%%% Set parameters
clear all;
%close all;
%clc
rng('default');

%%%%%%%%%%%%%%%%%%%%%%%%%      Calibration of Parameters      %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% Macro parameters
beta=0.97^(1/52);
chi = 1; %%% labor preference parameter

b = 6; %%%% price elasticity of demand
mu = b/(b-1); %markup for RE firm with full information

ambi_measure = 0.05; % measure of ambiguous firm
dampen = 0.1; % dampen factor


%%%%%%%%%%%%%%%%%%%%% Exogenous shocks

%%%%% S_t
mu_s = 0.002/4.3333; %%% Average inflation (Vavra (transformed to weekly))

rho_s = 1;
sigma_eps_s = 0.0015;  %%% Vavra transformed to weekly: 0.0037*12/52; 

%%%%% A_t
rho_a = 0.91^(1/4.3333); %%% Vavra (transformed to weekly), Fernald: 0.97^(4/52), 
sigma_eps_a = 0.006*sqrt((1-0.91^24)/(1-0.91^2))*sqrt((1-rho_a^2)/(1-rho_a^104)); %%% Vavra's numbers transformed to weekly

%%%%% w_t
rho_w = 0.9;
sigma_eps_w = 0.097; 

%%%%% Z_it
sigma_z = 0.3372/0.6745;  %%% Calibrated based on prediction error from weekly regressions

%%%%% Menu cost parameter
f = 0.023; %%% menu cost

%%% Learning and Ambiguity Parameters
alpha = 1.96; 
b_h = -b;  %%%make sure b_h <0
b_l = b_h; %%% Slopes of prior tunnel 
phi = 0;
phi_cutoff =  200;  %%% Defines how many lags of observations are held in memory
reset_observed = 0; %%% 1 if reset shocks observed, 0 if not observed
n = 2;

psi = 0; 
sigma_x = sigma_z*sqrt(0.2); 
b_max = 3*b_h; 

T = 31; %%% time between price reviews

gamma_h = n*sigma_z;  %%% Top intercept of the prior tunnel
gamma_l = -n*sigma_z; %%% Bottom intercept of the prior tunnel 
midQ = (gamma_h+gamma_l)/2;   %%%%mid-intercept

%%% Price grid (as a choice variable)
p_nodes = 2e3;
yref_nodes = 101;
n_y = 3; %%% Std deviation coverage on the grid for demand shocks; Needed for computing expectations below

%%% Length of simulation

T_sims = 3;
num_firms = 100; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%        Simulating a time path         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Initial Price History

p0 = -0.0225;
pagg0 = 0;%log(mu*chi) + mu_s;
y0_perc = [50];
n0 = 0.001;

% Grids point for Lengdure-Gauss integration
y0 = NaN(size(p0));
for i = 1:length(p0)
    y_grid = lgwt(yref_nodes, midQ+b_l*(p0(i) - pagg0(i))-n_y*sigma_z, midQ+b_h*(p0(i) - pagg0(i))+n_y*sigma_z);  %%% The signals are noisy because of z_it.
    y0(i) = y_grid(round(prctile(1:yref_nodes,y0_perc(i)))');
end

sigma_w = sqrt(sigma_eps_w^2/(1-rho_w^2));

%%% Initial Exogenous states
s0 = mu_s;
a0 = 0;
w0 = 0;

params = [midQ,b_h, sigma_eps_s, sigma_eps_a,sigma_eps_w, sigma_z, mu_s, rho_s, rho_a, rho_w, b, chi, b_l, gamma_h, gamma_l, alpha,f, mu, phi, p_nodes,pagg0,T,psi, sigma_x, phi_cutoff];

init_conds_ex =[s0,a0,w0];

eps_a = sigma_eps_a*normrnd(0,1,T_sims,1);
eps_s = sigma_eps_s*normrnd(0,1,T_sims,1);
eps_w_mat = sigma_eps_w*normrnd(0,1,T_sims,num_firms);
eps_z_mat = sigma_z*normrnd(0,1,T_sims,num_firms);

pmax_sims = NaN(T_sims, num_firms); 
pmax_menuc_sims = NaN(T_sims, num_firms); 
rmax_sims = NaN(T_sims, num_firms);
y_hist_sims = NaN(T_sims+1, num_firms);

%% simulate aggreagte process
s_sims(1) = mu_s + rho_s*mu_s + eps_s(1);
a_sims(1) = rho_a*a0 + eps_a(1);
for t = 2:T_sims
	s_sims(t) = mu_s + rho_s*s_sims(t-1) + eps_s(t);
	a_sims(t) = rho_a*a_sims(t-1) + eps_a(t);
end

%% Use flexible aggregate price as initial guess
p_agg_sims = log((b/(b-1))*chi*exp(s_sims-a_sims + 0.5*(sigma_z^2/(1-b) + (1-b)*sigma_w^2)));

diff = 10;
while diff > 1e-10
%% Compute flexbible firm price series
c_agg_sims = s_sims-p_agg_sims;
Pflex_sims = (b/(b-1))*chi*exp(s_sims-a_sims + 0.5*(sigma_z^2/(1-b) + (1-b)*sigma_w^2));
Yflex_sims = exp(c_agg_sims-b*(log(Pflex_sims)-p_agg_sims));

%% Compute each ambiguous firm's price and demand path 
tic
parfor ii = 1:num_firms
    
    eps_w = eps_w_mat(:,ii);
    eps_z = eps_z_mat(:,ii); 

    
    [y_hist_sims_temp,rmax_sims_temp,pflex_sims_temp,bound_hits,a_sims_temp,z_sims_temp,s_sims_temp,w_sims_temp,pjs_sims_temp] = sim_path_gp(init_conds_ex,p0,n0,y0,T_sims,eps_a,eps_s,eps_w,eps_z,p_agg_sims,params);
    
    pmax_sims(:,ii) = rmax_sims_temp + pjs_sims_temp; 
    rmax_sims(:,ii) = rmax_sims_temp; 
	y_hist_sims(:,ii) = y_hist_sims_temp;
    
    %bound_hits
end
toc

%% Find resulted ambiguous price component
ambi_price = ambi_measure*nanmean(exp(pmax_sims+y_hist_sims(2:end,:)),2)./exp(c_agg_sims)';
flex_price = (1-ambi_measure)*(Pflex_sims.*Yflex_sims./exp(c_agg_sims))';
Y_hist_sims = exp(y_hist_sims(2:end,:));
Yambi_sims = (nanmean(Y_hist_sims.^((b-1)/b),2)).^(b/(b-1));
Y_agg_sims = ambi_measure*Yambi_sims'+(1-ambi_measure)*Yflex_sims;
y_agg_sims = log(Y_agg_sims);
excess_supply = y_agg_sims-c_agg_sims;

% p_agg_sims_temp = log(ambi_price+flex_price);
% p_agg_sims_new = dampen*p_agg_sims_temp' + (1-dampen)*p_agg_sims;

step = (abs(excess_supply));
p_agg_sims_new = (1-sign(excess_supply).*step).*p_agg_sims;

diff = norm(p_agg_sims_new-p_agg_sims,2);
disp(diff);
disp(excess_supply);
p_agg_sims = p_agg_sims_new;

end



