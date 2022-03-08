%--------------------------------------------
%--------------------------------------------
% File to perform moment matching on DSGE model
%--------------------------------------------
% Jacob Warren
% jacobwar@sas.upenn.edu
%--------------------------------------------
% 5/22/2015
%--------------------------------------------
%--------------------------------------------

clc, clear, close all

%directory with additional functions:
addpath([pwd, '/functions']);

%path to save figures
picPath = [pwd, '/figures/moments/'];

% load DSGE model parameters
theta = parameters;


%% Moment Matching

rng(101)                % Set the seed for reproducibility
numReps     = 10;      % Number of simulations
sampleSizes = [80,200]; %Sample sizes to consider

zeta_hat      = zeros(numReps,length(sampleSizes));    %matrix to store zeta_hats
zeta_hat_dsge = zeros(numReps,length(sampleSizes));    %matrix to store optimal zetas
zeta_grid     = linspace(.2,.95, 50);   %% vary zetaP across the grid
N_E = 20;                    % Number of datasets to compute expected moments

% Loop over sample sizes
for n = 1:length(sampleSizes)
    
    N = sampleSizes(n);
    % Loop over Monte Carlo repetitions    
    
    parfor jj = 1:numReps
        % set the random seed
        rng(jj*N)

        % simulate "actual" data from DSGE 
        [Y_sim, s_sim] = DSGE_simulate(theta, N, 100);        
        Y_true = Y_sim(:,[1,3]);  % match only on output and inflation

        % estimate VAR(2) on "actual" data 
        p = 2;             
        Yp = Y_true(1:length(Y_true)-p,:);
        Xp = Y_true(2:length(Y_true)-(p-1),:);
        for j = 2:p
            Xp = [Xp, Y_true(j+1:length(Y_true)-(p-j),:)];
        end
        Xp = [ones(length(Xp),1),Xp];
        
        PhiHat_actual = (Xp'*Xp)\(Xp'*Yp);
        sigmaHat_actual = 1/length(Yp)*(Yp - Xp*PhiHat_actual)'*(Yp - Xp*PhiHat_actual);
        
        % Now construct the simulation-based MD estimator
        
        % use optimal weight matrix, computed from "true" data
        W = kron(inv(sigmaHat_actual),Xp'*Xp);          
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Objective function 1: use simulation to compute expected value
        % of VAR estimator
        %
        % use Q to store values of obj functions for different zetas
        Q = zeros(length(zeta_grid),1);
        
        % set theta_new to "true" theta and then only vary zeta_p
        theta_new = theta;
        
        % loop over zeta_grid
        for z = 1:length(zeta_grid)
            
            % update zeta_p
            theta_new(5) = zeta_grid(z);
            
            % initialize storage for VAR estimates
            PhiHat_sim = zeros(size(PhiHat_actual));
            
            % use the same random numbers across different zeta_p values
            % to decrease monte carlo error
            rng(jj + 1e5)
            
            for n_e = 1:N_E
                
                [Y_sim, s_sim] = DSGE_simulate(theta_new, N, 100);
                
                Y = Y_sim(:,[1,3]);
                
                Yp = Y(1:length(Y)-p,:);
                Xp = Y(2:length(Y)-(p-1),:);
                for j = 2:p
                    Xp = [Xp, Y(j+1:length(Y)-(p-j),:)];
                end
                Xp = [ones(length(Xp),1),Xp];
                
                PhiHat = (Xp'*Xp)\(Xp'*Yp);
                
                PhiHat_sim = PhiHat_sim + 1/N_E * PhiHat;
            end
            
            err = PhiHat_actual - PhiHat_sim;
            Q(z) = err(:)' * W * err(:);
        end
        
        [~,minLoc] = min(Q);
        
        zeta_hat(jj,n) = zeta_grid(minLoc);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Objective function 2: use DSGE population moments to construct
        % VAR estimates

        % DSGE implied VAR coefficient
        
        theta_new = theta;
        
        for z = 1:length(zeta_grid)
        
            zeta = zeta_grid(z);
            theta_new(5) = zeta;
            [phiHat_dsge, ~, ~] = DSGE_VAR_2(theta_new, 2);
            PhiHat_sim = phiHat_dsge(:,1:2);

            err = PhiHat_actual - PhiHat_sim;
            Q(z) = err(:)' * W * err(:);
        end
        
        [~,minLoc] = min(Q);
        
        zeta_hat_dsge(jj,n) = zeta_grid(minLoc);
        
    end
    
end

% Generate some plots

zetaP = theta(5);

[d, grid] = ksdensity(zeta_hat(:,1));
plot(grid,d, 'linewidth',4, 'Color','blue', 'linestyle',':')
hold on

[d, grid] = ksdensity(zeta_hat(:,2));
plot(grid,d, 'linewidth',4, 'Color','black', 'linestyle','--')
plot([zetaP zetaP], [0 max(d)+10],...
        'Color','red','LineWidth',2.5 ), hold off
ylim([0,8])
xlim([0,1])
set(gca, 'Xtick',0:.2:1)
set(gca,'fontsize',20,'fontweight','demi')
box off
print('-dpng',[picPath, 'MM'])

%Blue (dotted) is sample size 80, black (dashed) is sample size 200, red
%(solid) is true value

hold off
figure(2)
[d, grid] = ksdensity(zeta_hat_dsge(:,1));
plot(grid,d, 'linewidth',4, 'Color','blue', 'linestyle',':')
hold on

[d, grid] = ksdensity(zeta_hat_dsge(:,2));
plot(grid,d, 'linewidth',4, 'Color','black', 'linestyle','--')

plot([zetaP zetaP], [0 max(d)+10],...
        'Color','red','LineWidth',2.5 ), hold off
ylim([0,8])
xlim([0,1])
set(gca,'fontsize',20,'fontweight','demi')
set(gca, 'Xtick',0:.2:1)
box off
print('-dpng',[picPath, 'MM_dsge'])

%Blue (dotted) is sample size 80, black (dashed) is sample size 200, red
%(solid) is true value
