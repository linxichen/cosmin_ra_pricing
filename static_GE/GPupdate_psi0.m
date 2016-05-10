%  This fucntion takes a vector of prices, and evaluates the max-min
%  log-demand, using GP priors
%%%% GPUpdate -- a vector of worst-case expected demand at the grid of prices r 
%%%% PostVar -- just a scalar 
%%%% p0bs, n0bs, y0bs -- column vectors, with length equal to the number of
%%%% signals in the firm's memory

function [GPupdate, PostVar] = GPupdate_psi0(p,pObs,yObs,nObs,gamma_h,b_h,gamma_l,b_l,sigma_z,psi, sigma_x,pGrid)%ldmm0(x,P,Q,q_h,s_h,q_l,s_l,alpha,sigma,pGrid)
% p is the set of points you want to evaluate worst-case expected demand
% pObs, yObs, nObs are the position, average value and number of times you've seen
% signals
% pGrid -- is the grid

p_h = max(pGrid);
p_l = min(pGrid);

%pObs, nObs, yObs must be nX1 vectors.
%to check that input data has the right format:
%checks:

[n1,k1] = size(pObs);
[n2,k2] = size(yObs);
[n3,k3] = size(nObs);

%%
data  = sortrows([pObs yObs nObs]); %sorted data for price and quantity
unique_p  = unique(data(:,1));     %%%% Collect the unique prices in your memory
unique_p = unique_p(not(isnan(unique_p)));
data_unique = NaN(length(unique_p),3);

%%% Creates a condensed data matrix -- with rows for unique prices only 
for i=1:length(unique_p)
    datat = data(data(:,1)==unique_p(i,1),:);
    nt    = sum(datat(:,3));
    avt   = sum(datat(:,2).*datat(:,3))/nt;
    data_unique(i,:) = [datat(1) avt nt];   
end

data = data_unique;


pU    = data(:,1); %sorted and unique prices
yM    = data(:,2); %sorted mean for quantities
ySE    = sigma_z./sqrt(data(:,3)); %sorted standard-deviation for quantities

n = length(pU);  %%% number of unique signals seen

%%%%% this part gets rid of the data that are outside the grid limits
ind = pU>=p_l & pU<=p_h;
pU  = pU(ind);
yM  = yM(ind);
ySE  = ySE(ind);


%%% Compute alpha -- the vector of weights for the signals where
%%% E(x(p) | y ) = m(p) + alpha'(y - m)
%%% Start with expressing everything as multivariate normal
%%% Just applying standard conditional expectation formla (Wikipedia)

Sigma22 = sigma_x^2 + diag(ones(n,1).*(ySE.^2));
%%%% alpha = Cov / Var.  Cov = sigma_x^2
alpha = repmat(sigma_x^2,1,length(pU))/Sigma22;   %%%% Kalman gain. This a matrix length(r) X length(p_unique) 

PostVar = sigma_x^2 - alpha*repmat(sigma_x^2,1,length(pU))';   %%% Standard formula for posterior variance 

if sum(alpha,2) >1
    error('GPupdate error -> alphas sum to more than 1. We have an issue with worst case prior algorithm')
end

if PostVar < 0
    error('Posterior Variance comes out negative');
end

%%% Compute worst-case demand

GPupdate = NaN(1,length(p));

GPupdate(p>pU(end)) = gamma_l + b_h*p(p>pU(end)) + alpha*(yM - (gamma_h + b_h*pU));

for i=1:n-1
    ind_below = 1:n-i;
    ind_above = n-i+1:n;
    yhat_low  = alpha(1,ind_below)*(yM(ind_below) - (gamma_h + b_h*pU(ind_below)));   %%% Update from signals at lower reference prices 
    yhat_high = alpha(1,ind_above)*(bsxfun(@minus,yM(ind_above), bsxfun(@min,gamma_l +  b_h*p(p> pU(end-i) & p<= pU(end-i+1)),gamma_h + b_h*pU(ind_above))));   %%% Updates from signals at higher reference prices; 
    GPupdate(p> pU(end-i) & p<= pU(end-i+1)) = (gamma_l + b_h*p(p> pU(end-i) & p<= pU(end-i+1))) + yhat_low + yhat_high;  %%%% Update -- prior (at low bound) + updates coming form signals at lower prices + updates coming from signals at higher prices
end

GPupdate(p<= pU(1)) = (gamma_l + b_h*p(p<=pU(1))) + alpha*(bsxfun(@minus,yM, bsxfun(@min,gamma_l +  b_h*p(p<=pU(1)),gamma_h + b_h*pU)));


end
