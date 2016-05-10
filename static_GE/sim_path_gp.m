%%%%
%%%% Simulates a time-series of prices for ambiguity, flexible and menu
%%%% cost firms.
%%%% Version 2: Uses separate r and p grids. r grid for ambiguity firm, p
%%%% grid for menu cost 

function [y_hist_sims,rmax_sims,pflex_sims,bound_hits,a_sims,z_sims,s_sims,w_sims,pjs_sims] = sim_path_gp(init_conds_ex,p0,n0,y0,T_sims,eps_a,eps_s,eps_w,eps_z,p_agg_sims,params)

%%% Setting parameters
midQ = params(1);
b_h  = params(2);
sigma_eps_s = params(3);
sigma_eps_a = params(4);
sigma_eps_w = params(5);
sigma_z     = params(6);
mu_s        = params(7);
rho_s       = params(8);
rho_a       = params(9);
rho_w       = params(10);
b           = params(11);
chi         = params(12);
b_l         = params(13);
gamma_h     = params(14);
gamma_l     = params(15);
alpha       = params(16);
f           = params(17);
mu          = params(18);
phi         = params(19);
r_nodes     = params(20);
pjs0        = params(21);
T           = params(22);
psi         = params(23);
sigma_x     = params(24);
phi_crit    = params(25); 

%%%% Initial conditions
s0 = init_conds_ex(1);
a0 = init_conds_ex(2);
w0 = init_conds_ex(3);

sigma_w = sqrt(sigma_eps_w^2/(1-rho_w^2));


%%%%% Set price grid 
rlower_bound_adj = 7;
rupper_bound_adj = 7;

temp = sqrt(ceil(T/2)*sigma_eps_s^2 + sigma_eps_a^2*(1- rho_a^(2*ceil(T/2)))/(1-rho_a^2) + sigma_w^2);

r_grid = linspace( 0*ceil(T/2)*mu_s - 0.5*(sigma_z^2/(1-b) + (1-b)*sigma_w^2) - rlower_bound_adj*temp, ...
    0*ceil(T/2)*mu_s - 0.5*(sigma_z^2/(1-b) + (1-b)*sigma_w^2) + rupper_bound_adj*temp, r_nodes);

r = sort([r_grid,(p0-pjs0)']);
r_step = r(2) - r(1);
r_step_crit = 2;

%%%% Exogenous states  
s_sims = NaN(T_sims,1);  %%% Money 
a_sims = NaN(T_sims,1);  %%% Aggregate TFP
w_sims = NaN(T_sims,1);  %%% Idiosyncratic TFP

% p_agg_sims = NaN(T_sims,1);  %%% Aggregate price
c_agg_sims = NaN(T_sims,1);  %%% Aggregate quantity
pjs_sims   = NaN(T_sims,1);  %%%% Industry-level price (last seen) 

%%%% Define some holding variables for optimal stuff (optimized profits,
%%%% optimal prices etc. ...)
gmax_sims     = NaN(T_sims,1);  %%% Maximized profit
rmax_sims     = NaN(T_sims,1);  %%% Optimal price choice
rmax_ind_sims = NaN(T_sims,1);  %%% Grid index of the optimal price choice

pflex_sims = NaN(T_sims,1);    %%% Time series of optimal flexible price

r_hist_sims = NaN(T_sims + length(p0),1);   %%% History of prices the firm has seen
y_hist_sims = NaN(T_sims + length(p0),1);   %%% History of quantity signals 
n_hist_sims = NaN(T_sims + length(p0),1);   %%% History of number of observations

r_hist_sims(1:length(p0)) = p0 - pjs0;    %%% Initial price history
y_hist_sims(1:length(y0)) = y0;           %%% Value of initial signals
n_hist_sims(1:length(n0)) = n0;

%%% Helper variables that count how many times you hit the price grid
%%% boundary
rmin_count = 0;
rmax_count = 0;


%%% First loop iteration
t=1;

%%%% Exogenous state variables 
z_sims  = eps_z; 
s_sims(1) = mu_s + rho_s*s0 + eps_s(1);
a_sims(1) = rho_a*a0 + eps_a(1);
w_sims(1) = rho_w*w0 + eps_w(1);

r_hist = r_hist_sims(1:length(p0));   %%% Operational history 
y_hist = y_hist_sims(1:length(p0));   %%% Operational history of signals
n_hist = n_hist_sims(1:length(n0));

%%% Aggregates
% p_agg_sims(t) = log((b/(b-1))*chi*exp(s_sims(t)-a_sims(t) + 0.5*(sigma_z^2/(1-b) + (1-b)*sigma_w^2)));   %%% Aggregate price
% c_agg_sims(t) = log(((b-1)/b)*(1/chi)*exp(a_sims(t) - 0.5*(sigma_z^2/(1-b) + (1-b)*sigma_w^2)));        %%% Aggregate quantity
c_agg_sims(t) = s_sims(t)-p_agg_sims(t);        %%% Aggregate quantity

pjs_sims(t) = pjs0;    %%% Industry level price 

%%% Worst-Case Demand Expectation

[qDemand, sigmaxsqr_hat] = GPupdate_psi0(r,r_hist,y_hist,n_hist ,gamma_h,b_h,gamma_l,b_l,sigma_z,psi,sigma_x,r);

%%% Ambiguity averse firm's problem
g = ((exp(r + pjs_sims(t))/exp(p_agg_sims(t)))-chi*exp(s_sims(t))/(exp(p_agg_sims(t) + a_sims(t) + w_sims(t)))).*exp(0.5*(sigma_z^2 + sigmaxsqr_hat) + b*(p_agg_sims(t) - pjs_sims(t)) + c_agg_sims(t) + qDemand); %%% Calculates the profit function at each point in the price grid.

[g_max_iter, rmax_ind_iter] = max(g);

%%% next two lines just check if you hit a boundary
rmin_count = rmin_count + (rmax_ind_iter == 1);
rmax_count = rmax_count + (rmax_ind_iter == length(r));

gmax_sims(t) = g_max_iter;   %%% Save maximized profits
rmax_sims(t) = r(rmax_ind_iter);  %%% Optimal price

r_hist_sims(length(p0) + t) = rmax_sims(t);
y_hist_sims(length(p0) + t) = midQ - b*rmax_sims(t)+ z_sims(t);    %%% Actual realization of demand (under true DGP)
n_hist_sims(1: length(p0) + t-1) = n_hist_sims(1: length(p0) + t-1)*(1-phi);   %%% phi is discount rate. Right now = 0 
n_hist_sims(length(p0) + t) = (1-phi);

padj_ind = 0;

for t = 2:T_sims
    s_sims(t) = mu_s + rho_s*s_sims(t-1) + eps_s(t);
    a_sims(t) = rho_a*a_sims(t-1) + eps_a(t);
    w_sims(t) = rho_w*w_sims(t-1) + eps_w(t);
    
    r_hist = r_hist_sims(max(length(p0) +  t-phi_crit,1):length(p0) + t-1);   %%% Pick out last 200 periods 
    y_hist = y_hist_sims(max(length(p0) +  t-phi_crit,1):length(p0) + t-1);
    n_hist = n_hist_sims(max(length(p0) +  t-phi_crit,1):length(p0) + t-1);
    
    %%% Aggregates
    % p_agg_sims(t) = log((b/(b-1))*chi*exp(s_sims(t)-a_sims(t) + 0.5*(sigma_z^2/(1-b) + (1-b)*sigma_w^2)));
    % c_agg_sims(t) = log(((b-1)/b)*(1/chi)*exp(a_sims(t) - 0.5*(sigma_z^2/(1-b) + (1-b)*sigma_w^2)));
    c_agg_sims(t) = s_sims(t)-p_agg_sims(t);
	
    pjs_sims(t)   =  (mod(t,T) == 0)*p_agg_sims(t) + (1 - (mod(t,T) == 0))*pjs_sims(t-1);   %%%%% This line must change
    
    %%% Instead it should be something like (rand < lambda)*(p_agg_sims(t)
    %%% + noise) + (1 - (rand < lambda))*pjs_sims(t-1) -- see page 6 of GE
    %%% note
    
    
    %%% Next few lines re-center grids every once in a while 
    p_star = log(mu*chi) + (s_sims(t)-a_sims(t)- w_sims(t));
    
    if  (min(abs(p_star - (r(1) + pjs_sims(t))),abs(p_star - (r(end) + pjs_sims(t)))) < r_step_crit*temp) && (mod(t,T) == 0)
        r_grid = linspace( min([p_star - pjs_sims(t) - rlower_bound_adj*temp;r_hist]), ...
            max([p_star - pjs_sims(t) + rupper_bound_adj*temp;r_hist]), r_nodes-1);

        r = sort([r_grid,r_hist']);
        padj_ind = padj_ind +1;
    end
    
    %%% Worst-case Demand Expectations
    [qDemand, sigmaxsqr_hat] = GPupdate_psi0(r,r_hist,y_hist,n_hist ,gamma_h,b_h,gamma_l,b_l,sigma_z,psi,sigma_x,r);
    
    %%% Ambiguity averse firm's problem
    g = ((exp(r + pjs_sims(t))/exp(p_agg_sims(t)))-chi*exp(s_sims(t))/(exp(p_agg_sims(t) + a_sims(t) + w_sims(t)))).*exp(0.5*(sigma_z^2 + sigmaxsqr_hat) + b*(p_agg_sims(t) - pjs_sims(t)) + c_agg_sims(t) + qDemand); %%% Calculates the profit function at each point in the price grid.

    [g_max_iter, rmax_ind_iter] = max(g);
    
    rmin_count = rmin_count + (rmax_ind_iter == 1);
    rmax_count = rmax_count + (rmax_ind_iter == length(r));
    
    gmax_sims(t) = g_max_iter;
    rmax_sims(t) = r(rmax_ind_iter);
    
    r_hist_sims(length(p0) + t) = rmax_sims(t);
    y_hist_sims(length(p0) + t) = midQ - b*rmax_sims(t) + z_sims(t);
    n_hist_sims(1: length(p0) + t-1) = n_hist_sims(1: length(p0) + t-1)*(1-phi);
    n_hist_sims(length(p0) + t) = (1-phi);
    
    t;
end


% bound_hits = max([rmax_count,rmin_count,pmin_count_menuc]);
bound_hits = 0;

end