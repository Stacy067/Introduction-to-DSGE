%--------------------------------------------
%--------------------------------------------
% File to perform MLE of DSGE model
%--------------------------------------------
% Jacob Warren
% jacobwar@sas.upenn.edu
%--------------------------------------------
% 11/19/2015
%--------------------------------------------
%--------------------------------------------

clc, clear, close all

%add helper functions:
addpath([pwd, '/functions']);

%path to save pictures
picPath = [pwd, '/figures'];


%% Sampling Distribution of MLE

rng(101)                                        % Set the seed for reproducibility
numReps = 100;                                  % Number of simulations
sampleSizes = [80,200];                         % Sample sizes to consider
zeta_mle = zeros(numReps,length(sampleSizes));  % matrix to store optimal zetas
zeta_grid = linspace(.45,.95, 50);              % vary zetaP across the grid
theta = parameters;                             % load true parameters

parfor n = 1:length(sampleSizes)

    N = sampleSizes(n);
    rng(n*100)

    for jj = 1:numReps
        
        [Yhat, ~] = DSGE_simulate(theta, N, 100); % generate a sample from DSGE
       
        Q = zeros(length(zeta_grid),1);    % initialize objective function
        theta_new = theta;                 % fix other parameters at true value
        
        for z = 1:length(zeta_grid)  % discrete grid for evaluating the likelihood 
            theta_new(5) = zeta_grid(z); % replace true zeta by grid value            
            % use kalman filter to evaluate likelihood
            Sigma_u = zeros(4);
            alpha = .1;
            [~,~,~,~, liki] = kalman_wrapper(theta_new, Yhat, alpha, Sigma_u);
            Q(z) = sum(log(liki)); % sum up the log likelihood increments        
        end
        
        [~,maxLoc] = max(Q); % find maximum
        zeta_mle(jj,n) = zeta_grid(maxLoc);  % record mle
        
    end
end
zetaP = theta(5);

[d, grid] = ksdensity(zeta_mle(:,1));
plot(grid,d, 'linewidth',4, 'Color','blue', 'linestyle',':')
hold on

[d, grid] = ksdensity(zeta_mle(:,2));
plot(grid,d, 'linewidth',4, 'Color','black', 'linestyle','--')

plot([zetaP zetaP], [0 max(d)+10],...
    'Color','red','LineWidth',2.5 ), hold off
ylim([0,max(d)+1])
xlim([0,1])
set(gca, 'Xtick',0:.2:1)
set(gca,'fontsize',20,'fontweight','demi')
box off
print('-dpng',[picPath, 'samp_distr_mle'])

%Blue (dotted) is sample size 80, black (dashed) is sample size 200, red
%(solid) is true value


%% plot single likelihood cut:

theta = parameters; % load true parameters
zeta_grid = linspace(.45,.80, 50); %vary zetaP across the grid
Q = zeros(length(zeta_grid),1); %initialize objective function

[Yhat, s_true] = DSGE_simulate(theta, 200, 100); % simulate sample

for z = 1:length(zeta_grid)   
    theta_new    = theta;
    theta_new(5) = zeta_grid(z); % replace true zeta by grid value
    %kalman filter for likelihood
    Sigma_u = zeros(4);
    alpha = .1;
    [~,~,~,~, liki] = kalman_wrapper(theta_new, Yhat, alpha, Sigma_u);   
    Q(z) = sum(log(liki));
end

plot([theta(5) theta(5)], [0 max(Q)+20],'Color','red','LineWidth',2.5 )
hold on
plot(zeta_grid, Q, 'linewidth',4, 'Color','blue')

ylim([min(Q)-20,max(Q)+20])
set(gca, 'Xtick',0:.1:1)
set(gca,'fontsize',20,'fontweight','demi')

box off
print('-dpng',[picPath, 'likelihood'])

