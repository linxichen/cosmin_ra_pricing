function [z_grid, prob_z,z_cdf] = ar1_markov(std_e,rho_z,z_nodes) 
%%% Arguments 
%%% 1) std of shock 
%%% 2) Autocorrelation 
%%% 3) Number of grid nodes 

n_crit = 3; 

std_z = std_e/sqrt(1-rho_z^2); 
z_grid = NaN(z_nodes,1);

z_ints = norminv((1:z_nodes-1)/z_nodes,0,std_z);

z_grid(1) = -std_z*z_nodes*normpdf(z_ints(1)/std_z);
z_grid(2:end-1) = -std_z*z_nodes*(normpdf(z_ints(2:end)/std_z)-normpdf(z_ints(1:end-1)/std_z));
z_grid(end) = std_z*z_nodes*(normpdf(z_ints(end)/std_z));

%{
z_ints = norminv(normcdf(-n_crit,0,1)+((1:z_nodes-1)/z_nodes).*(1 - 2*normcdf(-n_crit,0,1)),0,std_z);
z_beg = norminv(normcdf(-n_crit,0,1),0,std_z); 
z_end = norminv(1-normcdf(-n_crit,0,1),0,std_z); 

z_grid(1) = -std_z*(z_nodes/(1-2*normcdf(-n_crit,0,1)))*(normpdf(z_ints(1)/std_z) - normpdf(z_beg/std_z)); 
z_grid(2:end-1) = -std_z*(z_nodes/(1-2*normcdf(-n_crit,0,1)))*(normpdf(z_ints(2:end)/std_z)-normpdf(z_ints(1:end-1)/std_z));
z_grid(end) = std_z*(z_nodes/(1-2*normcdf(-n_crit,0,1)))*(normpdf(z_ints(end)/std_z) - normpdf(z_end/std_z));


z_grid = linspace(-std_z*n_crit,std_z*n_crit,z_nodes); 
%}

prob_z = NaN(z_nodes,z_nodes);
quad_nodes = 101;
for i = 1:z_nodes
    if i ==1
        [temp_grid,w_temp] = lgwt(quad_nodes,z_ints(i)-5*std_z,z_ints(i));
    elseif i==z_nodes
        [temp_grid,w_temp] = lgwt(quad_nodes,z_ints(i-1),z_ints(i-1)+5*std_z);
    else
        [temp_grid,w_temp] = lgwt(quad_nodes,z_ints(i-1),z_ints(i));
    end
    
    for j = 1:z_nodes
        if j ==1
            integrand = normcdf((z_ints(j) - rho_z*temp_grid)/std_e,0,1);
        elseif j == z_nodes
            integrand = 1 - normcdf((z_ints(j-1) - rho_z*temp_grid)/std_e,0,1);
        else
            integrand = normcdf((z_ints(j) - rho_z*temp_grid)/std_e,0,1) - normcdf((z_ints(j-1) - rho_z*temp_grid)/std_e,0,1);
        end
        integrand = z_nodes*integrand.*normpdf(temp_grid,0,std_z);
        prob_z(i,j) = sum(integrand.*w_temp);
    end
end

z_cdf = NaN(z_nodes,z_nodes);
z_cdf(:,1) = prob_z(:,1);
for i = 2:z_nodes;
    z_cdf(:,i) = z_cdf(:,i-1) + prob_z(:,i);
end

end