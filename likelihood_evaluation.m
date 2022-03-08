%--------------------------------------------
%--------------------------------------------
% Evaluate DSGE model likelihood with Kalman filter
%
%--------------------------------------------
% Jacob Warren
% jacobwar@sas.upenn.edu
%--------------------------------------------
% 5/22/2015
%--------------------------------------------
%--------------------------------------------

clc, clear all, close all

%add helper functions:
addpath([pwd, '/functions']);

%path to save pictures
picPath = [pwd, '/figures/'];

% load DSGE model parameters
theta = parameters;

% solve model and generate state-space representation
[Phi1, Phi_eps, Psi0, Psi1] =  DSGE_soln_matrices(theta);



%% Kalman Filtering

%generate time series with 50 observations (100 used as burn-in to get to
%steady state
[Y_obs, s_true] = DSGE_simulate(theta, 50, 100);

%run the filter/smoother on Y(:,1:3)
Sigma_u = zeros(4);
alpha = .1;
[s_mean, P_mean, s_hi, s_lo, L] = kalman_wrapper(theta,Y_obs(:,1:3),alpha, Sigma_u);

%plot the true state and the filtered state
x = 1:50;                %helper values to print the background
X = [x,fliplr(x)];
for i = [1 3]
    figure(i)
    hold on
    Y_fill = [s_lo(:,i)',fliplr(s_hi(:,i)')];
    h = fill(X,Y_fill,[0 0 0] + .8);
    set(h, 'EdgeColor','None')
    plot(s_true(:,i), 'LineWidth',4,'Color','black', 'LineStyle',':')
    plot(s_mean(:,i),'Color','b','LineStyle','--','LineWidth',4)
    set(gca,'fontsize',20,'fontweight','demi')
    xlabel('Time','fontsize',22)
    
    xlim([0 50])
    if i==1
        ylim([-.06 .1])
    else
        ylim([-.03 .04])
    end
    box off
    print('-dpng',[picPath, 'filtered' num2str(i) 'from_3'])
    
    hold off
end

close all


%Now filter based on only Y(1)
[s_mean, P_mean, s_hi, s_lo, L] = kalman_wrapper(theta,Y_obs(:,1),alpha, Sigma_u);

%plot the true state and the filtered state
for i = [1 3]
    figure(i)
    hold on
    Y_fill = [s_lo(:,i)',fliplr(s_hi(:,i)')];
    h = fill(X,Y_fill,[0 0 0] + .8);
    set(h, 'EdgeColor','None')
    plot(s_true(:,i), 'LineWidth',4,'Color','black', 'LineStyle',':')
    plot(s_mean(:,i),'Color','b','LineStyle','--','LineWidth',4)
    set(gca,'fontsize',20,'fontweight','demi')
    xlabel('Time','fontsize',22)
    
    xlim([0 50])
    if i==1
        ylim([-.06 .1])
    else
        ylim([-.03 .04])
    end
    box off
    print('-dpng',[picPath, 'filtered' num2str(i) 'from_1'])
    
    hold off
end

close all

%black dotted line is true state, blue dashed line is kalman filtered
%state. Grey band represents 90% credible bands