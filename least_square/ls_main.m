% ========================================================================
% Walkman: A Communication-Efficient Random-Walk Algorithm for Decentralized Optimization
%
% cited as: arXiv:1804.06568
%
% Copyright(c) 2019 Xianghui Mao, Kun Yuan, Yubin Hu, Yuantao Gu, Ali H. Sayed, Wotao Yin
% All Rights Reserved.
% ----------------------------------------------------------------------
% This code is to test Walkman's performance on solving decentralized least 
% squares problem,
%               minimize_x  \sum_{i=1}^n (||A_i x - b_i||^2 +\lambda ||x||^2).
%
% Version 1.0
% Written by Xianghui Mao (maoxh92@sina.com) 
%----------------------------------------------------------------------

clear;clc;close all;
addpath ../functions/
addpath ./functions_least_square/

%% setting random seed
seed=[2017];
fprintf('Seed = %d\n',seed);
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

%% generating the least squares problem
n = 50;                                             % number of agents
p = 3;                                              % dimensionality of optimization variable
lmd = 0;                                            % regularization parameter lambda 
A = randn(n,p);                                     % sensing matrix A = [A_1; A_2; ...; A_n]
x_init = randn(p, 1);                               % x_0
sigma =1e-2;                                        % sensing noise level
b = A*x_init +sigma*randn(n, 1);                    % sensing result b = [b_1; b_2; ...; b_n]
xstar = (inv((A'*A)./n + lmd.*eye(p))*(A'*b))./n;   % optimal solution
ystar = repmat(xstar',n,1);                         % extended optimal solution

%% generating graph topology
network_generator = 'radius'                        % connection type, possible choices: 'linear', 'ring', 'radius', 'complete'
if network_generator == 'radius'                    
    networkRadius = 15;                             % communication radius of each agent
end

switch network_generator
    case 'linear'
        edge1 = zeros(1,n);
        edge1(2) = 1/3;
        edge1(1) = 1/3;
        W = toeplitz(edge1);
        W(1, 1) = 2/3;
        W(n, n) = 2/3;
        N_nd = n;
        N_eg = n-1;
    case 'ring'
        edge1 = zeros(1,n);
        edge1(2) = 1/3;
        edge1(1) = 1/3;
        W = toeplitz(edge1);
        W(1, n) = 1/3;
        W(n, 1) = 1/3;
        N_nd = n;
        N_eg = n-1;
    case 'radius'
        [cmat,incimat,nnum, Coordinates]=NetworkGen(n,30,30,networkRadius);
        incimat2=incimat(:,1:2:end);
        V=incimat2';
        [N_eg,N_nd]=size(V);        % N_eg: number of edges
        L=V'*V;
        tau=max(diag(2*L));
        W=eye(n)-2*L/tau;           % gossip matrix
        V=V/sqrt(tau);
    case 'complete'
        W = ones(n, n)./n;
        N_nd = n;
        N_eg = (n^2-n)/2;
end

%% basic parameters about graph topology
L = eye(n)-W;                       % Laplacian matrix
Lsquare = L^2;
Wtilde = (W+eye(n))./2;             % possitive definite gossip matrix
boolAdjacency = W>0;
network_Density = 2*N_eg/(N_nd^2 - N_nd);
topologyName = sprintf('Network_Density%f.mat', network_Density);

NeighborCell = cell(n, 1);
Neighbor_noloop = cell(n, 1);
degreeVec = zeros(1,n);
degreeVec_noloop = zeros(1, n);
for nodeInd = 1:n
    NeighborCell{nodeInd} = find(W(nodeInd,:));
    degreeVec(nodeInd) = length(NeighborCell{nodeInd});
    Neighbor_noloop{nodeInd} = setdiff( NeighborCell{nodeInd}, nodeInd);
    degreeVec_noloop(nodeInd) = length(Neighbor_noloop{nodeInd});
end
D_noloop = diag(degreeVec_noloop);
A_noloop = (W - diag(diag(W)))>0;

%% algorithm parameters

% maximum iterations for different methods
maxCommunicat=1e5;
maxIteration =1e3;

maxIteration_walkmanb = 1e5;
maxIteration_rand_fixed = 1e6;
maxIteration_rand = 1e6;
maxIteration_ADMM = 5e3;
maxIteration_walkmana = 1e5;
maxIteration_extra = 5e3;
maxIteration_diffusion = 5e3;

% setting stepsizes for different methods
alphaRangeEXTRA = [0.1];%complete: 0.11;ring: 0.08;linear: 0.02; radius: 0.13; %opt for N =20:2
alphaRangeDiffusion = [0.08];%[0.2];%complete: 0.22;ring: 0.08;linear: 0.02;%radius: 0.13; [3:2:10];
alphaRangewadmm = [2.5];
alphaRangeADMM = [0.15];
alphaRangerand = [@(x)min(1e-2, 5/(x))];
alphaRangerand_fixed = [1e-4];
alphaRangeWalk2 = [5e-2];

alphaSizeArray = [length(alphaRangeEXTRA), length(alphaRangeDiffusion)...
    ,  length(alphaRangewadmm), length(alphaRangeADMM), length(alphaRangerand_fixed), length(alphaRangerand),...
    length(alphaRangeWalk2)];
alphaSize = max(alphaSizeArray);
x_error = cell(alphaSize, 7);                       % caching errors
time_mu = 1;                                        % time model: parameter of exponential distribution


%% --- Simulation ---
for alphaInd = 1:alphaSize
    % automatically deciding whether to run a method
    flagAlpha_extra = alphaInd <= alphaSizeArray(1);
    flagAlpha_diffusion = alphaInd <= alphaSizeArray(2);
    flagAlpha_walkmana = alphaInd <= alphaSizeArray(3);
    flagAlpha_ADMM = alphaInd <= alphaSizeArray(4);
    flagAlpha_rand_fixed = alphaInd <= alphaSizeArray(5);
    flagAlpha_rand = alphaInd <= alphaSizeArray(6);
    flagAlpha_walkmanb = alphaInd <= alphaSizeArray(7);
    
    %% Walkman applying gradients (Walkman(b))
    if flagAlpha_walkmanb
        %% preparation for the application of Walkman(b)
        dataInd_walkmanb = 1;                       % activated agent
        alpha_walkmanb = alphaRangeWalk2(alphaInd)
        
        %% initialization for Walkman(b)
        z_walkmanb = zeros(n, p);
        y_walkmanb = zeros(n, p);
        xbar_walkmanb = zeros(1, p);
        iteration_walkmanb = 0;
        time_walkmanb = zeros(1, maxIteration_walkmanb);
        x_error{alphaInd, 7} = zeros(1, maxIteration_walkmanb);
        
        while iteration_walkmanb < maxIteration_walkmanb
            iteration_walkmanb = iteration_walkmanb + 1;
            time_walkmanb(iteration_walkmanb) = exprnd(time_mu, 1);                 % communication time
            dataInd_walkmanb = Neighbor_noloop{dataInd_walkmanb}(randperm(degreeVec_noloop(dataInd_walkmanb), 1));            % walk to a neighboring agent
            g_now_walkmanb = (A(dataInd_walkmanb,:)'*(A(dataInd_walkmanb,:)*y_walkmanb(dataInd_walkmanb, :)'-b(dataInd_walkmanb)))' + lmd.* y_walkmanb(dataInd_walkmanb, :);        % local gradient evaluation
            ypre_walkmanb = y_walkmanb(dataInd_walkmanb, :);
            zpre_walkmanb = z_walkmanb(dataInd_walkmanb, :);
            y_walkmanb(dataInd_walkmanb, :) =  xbar_walkmanb + z_walkmanb(dataInd_walkmanb, :) - alpha_walkmanb.*g_now_walkmanb;            % updating local primal variable
            z_walkmanb(dataInd_walkmanb, :) = xbar_walkmanb - y_walkmanb(dataInd_walkmanb, :) + z_walkmanb(dataInd_walkmanb, :);            % updating local dual variable
            xbar_walkmanb = xbar_walkmanb + ((y_walkmanb(dataInd_walkmanb, :) - z_walkmanb(dataInd_walkmanb, :))- (ypre_walkmanb - zpre_walkmanb))./n;      % updating primal variable on token
            x_error{alphaInd, 7}(iteration_walkmanb)=  norm(y_walkmanb - ystar, 'fro')^2/n;                                                 % error evaluation
        end
        communication_walkmanb = [1:iteration_walkmanb];
        time_walkmanb = cumsum(time_walkmanb(1:iteration_walkmanb));
        x_error{alphaInd, 7} = x_error{alphaInd, 7}(1:iteration_walkmanb);
    end
    
    
    %% RW (random walk) with constant stepsize
    if flagAlpha_rand_fixed
        alpha_rand_fixed = alphaRangerand_fixed(alphaInd)
        dataInd_rand_fixed = 1;
        %% initialization for RW with constant stepsize
        x_error{alphaInd,5} = zeros(1, maxIteration_rand_fixed);
        time_rand_fixed = zeros(1, maxIteration_rand_fixed);
        iterate_rand_fixed = 0;
        x_rand_fixed = zeros(p, 1);
        
        while iterate_rand_fixed < maxIteration_rand_fixed
            iterate_rand_fixed = iterate_rand_fixed + 1;
            time_rand_fixed(iterate_rand_fixed) = time_rand_fixed(iterate_rand_fixed) + exprnd(time_mu);            % communication time
            dataInd_rand_fixed = Neighbor_noloop{dataInd_rand_fixed}(randperm(degreeVec_noloop(dataInd_rand_fixed),1));         % walk to a neighboring agent
            x_rand_fixed = x_rand_fixed - alpha_rand_fixed.*((A(dataInd_rand_fixed,:)'*(A(dataInd_rand_fixed,:)*x_rand_fixed - b(dataInd_rand_fixed))) + lmd.* x_rand_fixed);       % updating variable on token
            x_error{alphaInd,5}(iterate_rand_fixed) = x_error{alphaInd,5}(iterate_rand_fixed) + norm(x_rand_fixed-xstar,'fro')^2;           % error evaluation
        end
        communication_rand_fixed = [1:iterate_rand_fixed];
        time_rand_fixed = cumsum(time_rand_fixed(1:iterate_rand_fixed));
        x_error{alphaInd, 5} = x_error{alphaInd, 5}(1:iterate_rand_fixed);
    end
    
    %% RW (random walk) with decaying stepsize
    if flagAlpha_rand
        alpha_rand = alphaRangerand(alphaInd)
        %% initialization for RW (random walk) with decaying stepsize
        x_error{alphaInd,6} = zeros(1, maxIteration_rand);
        time_rand = zeros(1,maxIteration_rand);
        dataInd_rand = 1;
        iterate_rand = 0;
        x_rand = zeros(p, 1);
        
        while iterate_rand < maxIteration_rand
            iterate_rand = iterate_rand + 1;
            time_rand(iterate_rand) = time_rand(iterate_rand) + exprnd(time_mu);            % communication time
            dataInd_rand = Neighbor_noloop{dataInd_rand}(randperm(degreeVec_noloop(dataInd_rand),1));           % walk to a neighboring agent
            stepsize_rand = alphaRangerand(iterate_rand);                                                       % decay stepsize
            x_rand = x_rand - ((A(dataInd_rand,:)'*(A(dataInd_rand,:)*x_rand - b(dataInd_rand))) + lmd.* x_rand).*stepsize_rand;              % updating variable on token
            x_error{alphaInd,6}(iterate_rand) =  x_error{alphaInd,6}(iterate_rand) + norm(x_rand-xstar,'fro')^2;    % error evaluation    
        end
        communication_rand = [1:iterate_rand];
        time_rand = cumsum(time_rand(1:iterate_rand));
        x_error{alphaInd, 6} = x_error{alphaInd, 6}(1:iterate_rand);
    end
    
    %% D-ADMM
    if flagAlpha_ADMM
        %% preparation for the application of D-ADMM
        alpha_ADMM = alphaRangeADMM(alphaInd)
        gramUInv =  cell(n, 1);            % preparing for computing local proximal point
        gramU = cell(n, 1);
        for nodeInd = 1:n
            gramU{nodeInd} = A(nodeInd, :)'* A(nodeInd, :) + (2*alpha_ADMM*degreeVec_noloop(nodeInd)).*eye(p);
            gramUInv{nodeInd} = inv(gramU{nodeInd});
        end
        %% initialization
        x_ADMM = zeros(n, p);
        a_ADMM = zeros(n, p);
        iteration_ADMM = 0;
        time_ADMM = zeros(1, maxIteration_ADMM);
        x_error{alphaInd, 4} = zeros(1, maxIteration_ADMM);
        
        while iteration_ADMM < maxIteration_ADMM
            iteration_ADMM = iteration_ADMM + 1;
            temp_ADMM = alpha_ADMM.*(D_noloop + A_noloop)*x_ADMM - a_ADMM;
            x_ADMM = df_conjugate(A, gramUInv, b, temp_ADMM, n, p);
            a_ADMM = a_ADMM  + alpha_ADMM.*(D_noloop*x_ADMM - A_noloop*x_ADMM);
            x_error{alphaInd,4}(iteration_ADMM) = norm(x_ADMM-ystar,'fro')^2./n;
            time_inner = exprnd(time_mu, [2*N_eg,1]);           % time consumption of each communication
            maxtime1 = max(time_inner);                         % wait for the slowest one among all the 2*N_eg communications
            time_ADMM(iteration_ADMM) = maxtime1;
        end
        communication_ADMM = [1:iteration_ADMM].*(2*N_eg);
        time_ADMM = cumsum(time_ADMM(1:iteration_ADMM));
        x_error{alphaInd, 4} = x_error{alphaInd, 4}(1:iteration_ADMM);
    end
    
    %% Walkman applying proximal operators (Walkman(a))
    if flagAlpha_walkmana
        
        alpha_walkmana = alphaRangewadmm(alphaInd)
        %% preparation for the application of Walkman(a)
        gramInv = cell(n, 1);       % preparing for computing local proximal point
        for nodeInd = 1:n
            gramInv{nodeInd} = pinv(A(nodeInd, :)'*A(nodeInd, :)+ alpha_walkmana*eye(p));
        end
        %% initialization for Walkman(a)
        z_walkmana = zeros(n, p);
        y_walkmana = zeros(n, p);
        dataInd_walkmana = 1;
        xbar_walkmana = zeros(1, p);
        iteration_walkmana = 0;
        time_walkmana = zeros(1, maxIteration_walkmana);
        x_error{alphaInd, 3} = zeros(1, maxIteration_walkmana);
        
        while iteration_walkmana < maxIteration_walkmana
            iteration_walkmana = iteration_walkmana + 1;
            time_walkmana(iteration_walkmana) = exprnd(time_mu, 1);         % communication time
            dataInd_walkmana = Neighbor_noloop{dataInd_walkmana}(randperm(degreeVec_noloop(dataInd_walkmana), 1));      % walk to a neighboring agent
            ypre_walkmana = y_walkmana(dataInd_walkmana, :);
            zpre_walkmana = z_walkmana(dataInd_walkmana, :);
            y_walkmana(dataInd_walkmana, :) =  (gramInv{dataInd_walkmana}*(A(dataInd_walkmana, :)'*b(dataInd_walkmana) + alpha_walkmana.*(xbar_walkmana+z_walkmana(dataInd_walkmana, :))'))';  % updating local primal variable
            z_walkmana(dataInd_walkmana, :) = xbar_walkmana - y_walkmana(dataInd_walkmana, :) + z_walkmana(dataInd_walkmana, :);           % updating local dual variable
            xbar_walkmana = xbar_walkmana + ((y_walkmana(dataInd_walkmana, :) - z_walkmana(dataInd_walkmana, :))- (ypre_walkmana - zpre_walkmana))./n;      % updating primal variable on token
            x_error{alphaInd, 3}(iteration_walkmana)=  norm(y_walkmana - ystar, 'fro')^2/n;         % error evaluation
        end
        communication_walkmana = [1:iteration_walkmana];
        time_walkmana = cumsum(time_walkmana(1:iteration_walkmana));
        x_error{alphaInd, 3} = x_error{alphaInd, 3}(1:iteration_walkmana);
    end
    
    %% EXTRA
    if flagAlpha_extra
        alpha_extra = alphaRangeEXTRA(alphaInd)
        %% initialization for EXTRA
        x_extra = zeros(n, p);
        time_extra = zeros(1, maxIteration_extra);
        x_error{alphaInd, 1} = zeros(1, maxIteration_extra);
        %% the first iteration
        g_extra = df(A, b, x_extra, n, p, lmd);            % computing local gradients at the initial value
        x_extra_p1 = W*x_extra - alpha_extra.*g_extra;     % gossip and update
        iteration_extra = 1;
        time_inner = exprnd(time_mu, [2*N_eg,1]);          % communication time
        maxtime1 = max(time_inner);                        % wait for the slowest one among all the 2*N_eg communications
        time_extra(iteration_extra) = maxtime1;
        x_error{alphaInd,1}(iteration_extra) = norm(x_extra_p1-ystar,'fro')^2./n;       % error evaluation
        
        while  iteration_extra < maxIteration_extra
            g_extra_p1 = df(A, b, x_extra_p1, n, p, lmd);
            g_extra = df(A, b, x_extra, n, p, lmd);
            x_extra_p2 = Wtilde*(2.*x_extra_p1-x_extra) - alpha_extra.*g_extra_p1+ alpha_extra.*g_extra;
            x_extra = x_extra_p1;
            x_extra_p1 = x_extra_p2;
            iteration_extra = iteration_extra + 1;
            x_error{alphaInd,1}(iteration_extra) = norm(x_extra_p1-ystar,'fro')^2./n;
            time_inner = exprnd(time_mu, [2*N_eg,1]);
            maxtime1 = max(time_inner);
            time_extra(iteration_extra) = maxtime1;
        end
        communication_extra = [1:iteration_extra].*(2*N_eg);
        time_extra = cumsum(time_extra(1:iteration_extra));
        x_error{alphaInd, 1} = x_error{alphaInd, 1}(1:iteration_extra);
    end
    
    %% exact diffusion
    if flagAlpha_diffusion
        alpha_diffusion = alphaRangeDiffusion(alphaInd)
        %% initialization for exact diffusion
        x_diffusion = zeros(n,p);
        time_diffusion = zeros(1, maxIteration_diffusion);
        x_error{alphaInd, 2} = zeros(1, maxIteration_diffusion);
        %% the first iteration
        g_diffusion = df(A, b, x_diffusion, n, p, lmd);     % computing local gradients
        psi_diffusion = x_diffusion - alpha_diffusion.*g_diffusion;         % update
        x_diffusion_p1 = Wtilde * psi_diffusion;                            % gossip
        iteration_diffusion = 1;
        x_error{alphaInd,2}(iteration_diffusion) = norm(x_diffusion_p1-ystar,'fro')^2./n;   % error evaluation
        time_inner = exprnd(time_mu, [2*N_eg,1]);           % communication time
        maxtime1 = max(time_inner);                         % wait fot the slowest among all the 2*N_eg communications
        time_diffusion(iteration_diffusion) = maxtime1;
        
        while iteration_diffusion < maxIteration_diffusion
            g_diffusion_p1 = df(A, b, x_diffusion_p1, n, p, lmd);
            psi_diffusion_p1 = x_diffusion_p1 - alpha_diffusion.*g_diffusion_p1;
            phi_diffusion_p1 = x_diffusion_p1 + psi_diffusion_p1 - psi_diffusion;
            x_diffusion_p2 = Wtilde * phi_diffusion_p1;
            psi_diffusion = psi_diffusion_p1;
            x_diffusion_p1 = x_diffusion_p2;
            iteration_diffusion = iteration_diffusion + 1;
            x_error{alphaInd,2}(iteration_diffusion) = norm(x_diffusion_p1-ystar,'fro')^2./n;
            time_inner = exprnd(time_mu, [2*N_eg,1]);
            maxtime1 = max(time_inner);
            time_diffusion(iteration_diffusion) = maxtime1;
        end
        communication_diffusion = [1:iteration_diffusion].*(2*N_eg);
        time_diffusion = cumsum(time_diffusion(1:iteration_diffusion));
        x_error{alphaInd, 2} = x_error{alphaInd, 2}(1:iteration_diffusion);
    end
end

%% --- Plot figure ---
methodName = {'EXTRA          ';  'Exact Diffusion';'Walkman (11b)'; ...
    'D-ADMM        ';'RW Incremental\n(constant stepsize)'; 'RW Incremental\n(decaying stepsize)'; 'Walkman (11b'')'};
alphaRangeArray = {alphaRangeEXTRA; alphaRangeDiffusion; alphaRangewadmm;  alphaRangeADMM; alphaRangerand_fixed; alphaRangerand; alphaRangeWalk2};
CommunicateName = {'communication_extra','communication_diffusion',...
    'communication_walkmana', 'communication_ADMM', 'communication_rand_fixed', 'communication_rand','communication_walkmanb'};
TimeName = {'time_extra', 'time_diffusion', 'time_walkmana','time_ADMM', 'time_rand_fixed', 'time_rand', 'time_walkmanb'};
LineStyle = {'-', '--', ':', '-.', '-', '--', ':', '-', '-.', '--' ,'-', ':', '-.'};
Marker = {'o', '*', '+','s', 'd', 'p' };
C = linspecer(8);
fig = figure(1);
plotInd = 0
legendSet = cell(1,1);
timeLimit = min(time_walkmanb(end));
colorInd = [4,2,1,3,5,6,7];
plotIncludeZero = 1;                                        % switch whether plot the initial error
startError = norm(zeros(n,p) - ystar, 'fro')^2./n;          % initial error evaluation

for methodInd =[3,7,4,1,2,5,6]
    alphaRange = alphaRangeArray{methodInd};
    for alphaInd =1:max(1,alphaSizeArray(methodInd))
        plotInd = plotInd +1;
        plotnow = max((x_error{alphaInd,methodInd}),1e-16);
        communicationRange = eval(sprintf('%s',CommunicateName{methodInd}));
        timeRange = eval(sprintf('%s',TimeName{methodInd}));
        
        if plotIncludeZero
            timeRange = [0.001, timeRange];
            communicationRange = [0.001, communicationRange];
            plotnow = [startError, plotnow];
        end
        
        plotCommuteLength = max(find(communicationRange<=maxCommunicat));
        sub1 = subplot(1,2,1);
        loglog(communicationRange, plotnow,...
            'Color', C(colorInd(methodInd),:),'LineWidth', 3, 'LineStyle', LineStyle{plotInd});%colorInd(methodInd)
        hold on;
        xlim([1, 1e6]);
        ylim([1e-16, 1]);
        sub4 = subplot(1,2,2);
        plotCommuteLength = max(find(timeRange<=timeLimit));
        loglog(timeRange,plotnow, ...
            'Color', C(colorInd(methodInd),:),'LineWidth', 3, 'LineStyle', LineStyle{plotInd});
        xlim([1, 1e6]);
        ylim([1e-16, 1]);
        hold on;
        legendSet{plotInd} = [methodName{methodInd}];%, 'alpha=', sprintf('%.5f', alphaRangeArray{methodInd}(alphaInd))];%];%
    end
    
end
set(sub1,'YScale', 'log','Fontname', 'Times new Roman', 'FontSize', 18);
l1 = legend(sub1,legendSet, 'Location', 'SouthWest');
xlabel(sub1,'Communication Cost','Fontname', 'Times new Roman', 'FontSize', 18);

ylabel(sub1, '$\|{\scriptstyle{\bf\mathcal{Y}}}^k - {\scriptstyle{\bf\mathcal{Y}}}^*\|^2/n$','Interpreter', 'latex','Fontname', 'Times new Roman', 'FontSize', 18);
set(sub1,'Fontname', 'Times new Roman', 'FontSize', 18, 'xtick', [1e1,1e2,1e3,1e4,1e5, 1e6]);
set(sub4,'Fontname', 'Times new Roman', 'FontSize', 18, 'xtick', [1e1,1e2,1e3,1e4,1e5,1e6]);
xlabel(sub4,'Running Time','Fontname', 'Times new Roman', 'FontSize', 18);

