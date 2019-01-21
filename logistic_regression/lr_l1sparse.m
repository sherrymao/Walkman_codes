clear;clc;close all;
addpath ../functions/
addpath ./functions_logistic_regression/



seed=[2017];
fprintf('Seed = %d\n',seed);
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

maxCommunicat=5e4;
N=50;m=3;M=m;
maxIteration =5e3;

networkRadius = 15;


%% initialization
x_start=zeros(M,1);
y_start=zeros(M,1);

%% function settings
K=5;
lmd= 0;
beta = 0.1; 
% === Generate Data ===
U = randn(N*K, M);
w_0 = randn(M, 1);
d = ( rand(N*K,1)<(1./(1+exp(-U*w_0))) );
d = 2 * d- 1;

% === Calculate w_opt ===
% --- centralized gradient descent ---
w_c0 = zeros(M,1); w_c = w_c0; maxite_c = 300000; mu_c = 0.1;
for ite_c = 1:maxite_c
    prob = exp(-d.* (U * w_c) );
    prob = prob ./ (1+prob);
    grad = ( -U' * (d.* prob) )/(K*N) + lmd *  w_c;
    w_c = w_c - mu_c * grad;
    w_c = prox_l1(w_c, beta*mu_c);
end
x_true = w_c;
w0 = x_true;
w0_m = repmat(w0', N, 1);



eta = 1;
testing_par = 0;
alphaRangeWalk_ADMM =[1];%1; %[0.5, 0.8, 1, 1.5];%[1e-2, 0.1,0.2, 1, 2, 5, 10];%[0.05]%[1e-3, 5e-3,0.01:0.01:0.05];%[0.1];
alphaRange_pgextra = [0.5];%0.5;%[0.5, 0.8, 1,2];%[0.1, 0.2, 0.8, 5, 10];%0.8;%radius + unregularize: 0.8; radius + regularize: 0.6;%opt for N =20:2
alphaRange_rand = [0];%[0];
alphaRange_rand_fixed = [1e-3];%[1e-3];%[1e-3];
alphaRangeWalk2 = [0.5];%[0.1,0.2, 0.5, 1, 2, 5, 10];

alphaSizeArray = [length(alphaRangeWalk_ADMM), ...
    length(alphaRange_pgextra) ,  length(alphaRange_rand), length(alphaRange_rand_fixed),...
    length(alphaRangeWalk2)];
alphaSize = max(alphaSizeArray);
x_error = cell(alphaSize, 5);





%% --- Network Gen ---
network_generator = 'radius'
switch network_generator
    case 'linear'
        edge1 = zeros(1,N);
        edge1(2) = 1/3;
        edge1(1) = 1/3;
        W = toeplitz(edge1);
        W(1, 1) = 2/3;
        W(N, N) = 2/3;
        N_nd = N;
        N_eg = N-1;
    case 'ring'
        edge1 = zeros(1,N);
        edge1(2) = 1/3;
        edge1(1) = 1/3;
        W = toeplitz(edge1);
        W(1, N) = 1/3;
        W(N, 1) = 1/3;
        N_nd = N;
        N_eg = N-1;
    case 'radius'
        [cmat,incimat,nnum]=NetworkGen(N,30,30,networkRadius);
        incimat2=incimat(:,1:2:end);
        V=incimat2';[N_eg,N_nd]=size(V);
        L=V'*V;
        tau=max(diag(2*L));
        W=eye(N)-2*L/tau;
        V=V/sqrt(tau);A=2*V'*V;Vtran=V';
    case 'complete'
        W = ones(N, N)./N;
        N_nd = N;
        N_eg = (N^2-N)/2;
end


L = eye(N)-W;
Lsquare = L^2;
Wtilde = (W+eye(N))./2;
boolAdjacency = W>0;
DistMat = graphallshortestpaths(sparse(boolAdjacency));
netRadius = max(max(DistMat));
[~, masterInd] = min(sum(DistMat,1));
absorbCost = sum(DistMat(masterInd, :));
network_Density = 2*N_eg/(N_nd^2 - N_nd);
topologyName = sprintf('Network_Density%f.mat', network_Density);

NeighborCell = cell(N, 1);
Neighbor_noloop = cell(N, 1);
ProbabilityCell = cell(N, 1);
degreeVec = zeros(1,N);
degreeVec_noloop = zeros(1, N);
for nodeInd = 1:N
    NeighborCell{nodeInd} = find(W(nodeInd,:));
    ProbabilityCell{nodeInd} = W(nodeInd, NeighborCell{nodeInd});
    degreeVec(nodeInd) = length(NeighborCell{nodeInd});
    Neighbor_noloop{nodeInd} = setdiff( NeighborCell{nodeInd}, nodeInd);
    degreeVec_noloop(nodeInd) = length(Neighbor_noloop{nodeInd});
end
D_noloop = diag(degreeVec_noloop);
A_noloop = (W - diag(diag(W)))>0;
L_noloop = D_noloop - A_noloop;
L_plus = abs(L_noloop);



%% problem property
lambdaL = svd(L);
lambdaMax = lambdaL(1);
lambdaMin = lambdaL(N-1);
gammaL = lambdaMin/lambdaMax;
HessianDiag = zeros(N, 2);
for nodeInd = 1:N
    hessian_now = svd( U(nodeInd,:)'* U(nodeInd, :) + lmd.*eye(m));
    HessianDiag(nodeInd, :) = [max(hessian_now), min(hessian_now)];
end
strongly_convexity_local = min(HessianDiag(:, 2));
smoothness_local = max(HessianDiag(:, 1));
kappa_local = smoothness_local/strongly_convexity_local;


%% --- Simulation ---

svrg_innerLoop = 10;

%% --- match parameters ---
N_cache = K;
H = U;
label = d;
time_mu = 1;
for alphaInd = 1:alphaSize
    
    
    flagAlpha_walk_ADMM = alphaInd <= alphaSizeArray(1);
    flagAlpha_pgextra = alphaInd <= alphaSizeArray(2);
    flagAlpha_rand = alphaInd <= alphaSizeArray(3);
    flagAlpha_rand_fixed = alphaInd <= alphaSizeArray(4);
    flagAlpha_walkmanb = alphaInd <= alphaSizeArray(5);
   
    
    %% Walkman(b)
    if flagAlpha_walkmanb
        %% preparation for the application of Walkman(b)
        
        dataInd_walkmanb = 1;
        alpha_walkmanb = alphaRangeWalk2(alphaInd)
        %% initialization for Walkman(b)
        
        time_walkmanb = zeros(1, maxCommunicat);
        x_error{alphaInd, 5} = zeros(1, maxCommunicat);
        z_walkmanb = zeros(N, m);
        y_walkmanb = zeros(N, m);
        xbar_walkmanb = zeros(1, m);
        communicateRound_walkmanb = 0;
        iteration_walkmanb = 0;
        
        
        while communicateRound_walkmanb < maxCommunicat || iteration_walkmanb < maxIteration
            iteration_walkmanb = iteration_walkmanb + 1;
            time_walkmanb(iteration_walkmanb) = time_walkmanb(iteration_walkmanb)+ exprnd(time_mu);
            proposeInd = Neighbor_noloop{dataInd_walkmanb}(randperm(degreeVec_noloop(dataInd_walkmanb),1));
            randvar = rand(1);
            dataInd_walkmanb = proposeInd;
            
            ypre_walkmanb = y_walkmanb(dataInd_walkmanb, :);
            zpre_walkmanb = z_walkmanb(dataInd_walkmanb, :);
            x_walkmanb = prox_l1(xbar_walkmanb, beta*alpha_walkmanb);
            gnow = df_lr(H((dataInd_walkmanb-1)*N_cache+1:dataInd_walkmanb*N_cache , :),...
                label((dataInd_walkmanb-1)*N_cache+1:dataInd_walkmanb*N_cache), ypre_walkmanb, 1, N_cache, M, lmd);
            y_walkmanb(dataInd_walkmanb, :) =  x_walkmanb + z_walkmanb(dataInd_walkmanb, :) - gnow*alpha_walkmanb;
            z_walkmanb(dataInd_walkmanb, :) = x_walkmanb - y_walkmanb(dataInd_walkmanb, :) + z_walkmanb(dataInd_walkmanb, :);
            xbar_walkmanb = xbar_walkmanb + ((y_walkmanb(dataInd_walkmanb, :) - z_walkmanb(dataInd_walkmanb, :))- (ypre_walkmanb - zpre_walkmanb))./N;
            x_error{alphaInd, 5}(iteration_walkmanb)= x_error{alphaInd, 5}(iteration_walkmanb)+ norm(y_walkmanb - w0_m, 'fro')^2./N;
            
            communicateRound_walkmanb = communicateRound_walkmanb + 1;
        end
        communication_walkmanb = [1:iteration_walkmanb];
        time_walkmanb = cumsum(time_walkmanb(1:iteration_walkmanb));
        x_error{alphaInd, 5} = x_error{alphaInd, 5}(1:iteration_walkmanb);
    end
    
    
    %% Walkman (a)
    if flagAlpha_walk_ADMM
        %% preparation for the application of Walkman (a)
        seedset= [10:80:5000];
        dataInd_walk_ADMM = 1;
        alpha_walk_ADMM = alphaRangeWalk_ADMM(alphaInd)
        %% initialization for Walkman (a)
        nTrials = 1;
        time_walk_ADMM = zeros(1, maxCommunicat);
        x_error{alphaInd, 1} = zeros(1, maxCommunicat);
        for iTrial = 1:nTrials
            iTrial
            z_walk_ADMM = zeros(N, m);
            y_walk_ADMM = zeros(N, m);
            xbar_walk_ADMM = zeros(1, m);
            communicateRound_walk_ADMM = 0;
            iteration_walk_ADMM = 0;
            
            
            while communicateRound_walk_ADMM < maxCommunicat || iteration_walk_ADMM < maxIteration
                iteration_walk_ADMM = iteration_walk_ADMM + 1;
                time_walk_ADMM(iteration_walk_ADMM) = time_walk_ADMM(iteration_walk_ADMM)+ exprnd(time_mu);
                proposeInd = Neighbor_noloop{dataInd_walk_ADMM}(randperm(degreeVec_noloop(dataInd_walk_ADMM),1));
                randvar = rand(1);
                dataInd_walk_ADMM = proposeInd;
                
                ypre_walk_ADMM = y_walk_ADMM(dataInd_walk_ADMM, :);
                zpre_walk_ADMM = z_walk_ADMM(dataInd_walk_ADMM, :);
                x_walk_ADMM = prox_l1(xbar_walk_ADMM, beta*alpha_walk_ADMM);
                y_walk_ADMM(dataInd_walk_ADMM, :) =  logistic_regression_solver(H((dataInd_walk_ADMM-1)*N_cache+1:dataInd_walk_ADMM*N_cache , :) ,...
                    label((dataInd_walk_ADMM-1)*N_cache+1:dataInd_walk_ADMM*N_cache), (x_walk_ADMM + z_walk_ADMM(dataInd_walk_ADMM, :))', alpha_walk_ADMM);
                z_walk_ADMM(dataInd_walk_ADMM, :) = x_walk_ADMM - y_walk_ADMM(dataInd_walk_ADMM, :) + z_walk_ADMM(dataInd_walk_ADMM, :);
                xbar_walk_ADMM = xbar_walk_ADMM + ((y_walk_ADMM(dataInd_walk_ADMM, :) - z_walk_ADMM(dataInd_walk_ADMM, :))- (ypre_walk_ADMM - zpre_walk_ADMM))./N;
                x_error{alphaInd, 1}(iteration_walk_ADMM)= x_error{alphaInd, 1}(iteration_walk_ADMM)+ norm(y_walk_ADMM - w0_m, 'fro')^2./N;
                communicateRound_walk_ADMM = communicateRound_walk_ADMM + 1;
                
            end
        end
        communication_walk_ADMM = [1:iteration_walk_ADMM];
        time_walk_ADMM = cumsum(time_walk_ADMM(1:iteration_walk_ADMM)./nTrials);
        x_error{alphaInd, 1} = x_error{alphaInd, 1}(1:iteration_walk_ADMM)./nTrials;
    end
    
    
    
    %% P-EXTRA
    if flagAlpha_pgextra
        alpha_pgextra = alphaRange_pgextra(alphaInd)
        
        
        x_pgextra = zeros(N,m);
        g_pgextra =  df_lr(H, label, x_pgextra, N, N_cache, M, 0);
        x_pgextra_phalf = W*x_pgextra - alpha_pgextra.*g_pgextra;
        x_pgextra_p1 = prox_l1(x_pgextra_phalf, beta*alpha_pgextra);
        communication_pgextra = [];
        communicateRound_pgextra = 0;
        iteration_pgextra = 0;
        x_error{alphaInd, 2} = zeros(1, maxIteration);
        time_pgextra = zeros(1, maxIteration);
        while iteration_pgextra < maxIteration

            iteration_pgextra = iteration_pgextra + 1;
            g_pgextra_p1 = df_lr(H, label, x_pgextra_p1, N, N_cache, M, lmd);
            g_pgextra = df_lr(H, label, x_pgextra, N, N_cache, M, lmd);
            x_pgextra_p3half = W*x_pgextra_p1 + x_pgextra_phalf - Wtilde*x_pgextra ...
                - alpha_pgextra.*(g_pgextra_p1 - g_pgextra);
            x_pgextra_p2 = prox_l1(x_pgextra_p3half, beta*alpha_pgextra);
            x_pgextra = x_pgextra_p1;
            x_pgextra_phalf = x_pgextra_p3half;
            x_pgextra_p1 = x_pgextra_p2;
            x_error{alphaInd,2}(iteration_pgextra) = norm(x_pgextra_p1-w0_m,'fro')^2./N;
            time_inner = exprnd(time_mu, [2*N_eg,1]);
            maxtime1 = max(time_inner);
            time_pgextra(iteration_pgextra) = maxtime1;
            communicateRound_pgextra = communicateRound_pgextra + 2*N_eg;
            
        end
        communication_pgextra = [1:iteration_pgextra].*(2*N_eg);
        time_pgextra = cumsum(time_pgextra(1:iteration_pgextra));
        x_error{alphaInd, 2} = x_error{alphaInd, 2}(1:iteration_pgextra);
    end
    
    
    %% random walk decaying stepsize
    if flagAlpha_rand
        alpha_rand = alphaRange_rand(alphaInd);
        x_error{alphaInd,3} = zeros(1, 1e7);
        time_rand = zeros(1, 1e7);
        nTrials = 1;
        for iTrial = 1:nTrials
            iTrial
            RandStream.setGlobalStream(RandStream('mt19937ar','seed',seedset(iTrial)));
            dataInd_rand = randperm(N,1);
            communicateRound_rand = 0;
            iterate_rand = 0;
            x_rand = zeros(m, 1);
            while communicateRound_rand < maxCommunicat || iterate_rand < 1e7
                iterate_rand = iterate_rand + 1;
                time_rand(iterate_rand) = time_rand(iterate_rand) + exprnd(time_mu);
                proposeInd = Neighbor_noloop{dataInd_rand}(randperm(degreeVec_noloop(dataInd_rand),1));
                randvar = rand(1);
                dataInd_rand = proposeInd;
                grad_now = df_lr(H((dataInd_rand-1)*N_cache+1:dataInd_rand*N_cache , :),...
                    label((dataInd_rand-1)*N_cache+1:dataInd_rand*N_cache), x_rand', 1, N_cache, M, lmd);
                this_alpha = min(1e-2, 5/iterate_rand);
                x_rand = prox_l1(x_rand - grad_now'.*this_alpha, beta*this_alpha);
                x_error{alphaInd,3}(iterate_rand) =  x_error{alphaInd,3}(iterate_rand) + norm(x_rand-w0,'fro')^2;
                communicateRound_rand = communicateRound_rand + 1;
            end
        end
        communication_rand = [1:iterate_rand];
        time_rand = cumsum(time_rand(1:iterate_rand)./nTrials);
        x_error{alphaInd, 3} = x_error{alphaInd, 3}(1:iterate_rand)./nTrials;
    end
    
    
    
    %% random walk fixed stepsize
    if flagAlpha_rand_fixed
        seedset = [2017:100:10017];
        x_error{alphaInd,4} = zeros(1, 1e7);
        alpha_rand_fixed = alphaRange_rand_fixed(alphaInd)
        time_rand_fixed = zeros(1, 1e7);
        nTrials = 1;
        for iTrial = 1:nTrials
            iTrial
            communicateRound_rand_fixed= 0;
            iterate_rand_fixed = 0;
            RandStream.setGlobalStream(RandStream('mt19937ar','seed',seedset(iTrial)));
            dataInd_rand_fixed = randperm(N,1);
            x_rand_fixed = zeros(m, 1);
            while communicateRound_rand_fixed < maxCommunicat || iterate_rand_fixed < 1e7
                iterate_rand_fixed = iterate_rand_fixed + 1;
                time_rand_fixed(iterate_rand_fixed) = time_rand_fixed(iterate_rand_fixed) + exprnd(time_mu);
                proposeInd = Neighbor_noloop{dataInd_rand_fixed}(randperm(degreeVec_noloop(dataInd_rand_fixed), 1));
                randvar = rand(1);
                dataInd_rand_fixed = proposeInd;
                grad_now = df_lr(H((dataInd_rand_fixed-1)*N_cache+1:dataInd_rand_fixed*N_cache , :),...
                    label((dataInd_rand_fixed-1)*N_cache+1:dataInd_rand_fixed*N_cache), x_rand_fixed', 1, N_cache, M, lmd);
                x_rand_fixed = prox_l1(x_rand_fixed - alpha_rand_fixed.*grad_now', beta*alpha_rand_fixed);
                x_error{alphaInd,4}(iterate_rand_fixed) = x_error{alphaInd,4}(iterate_rand_fixed) + norm(x_rand_fixed-w0,'fro')^2;
                communicateRound_rand_fixed = communicateRound_rand_fixed + 1;
            end
        end
        communication_rand_fixed = [1:iterate_rand_fixed];
        time_rand_fixed = cumsum(time_rand_fixed(1:iterate_rand_fixed)./nTrials);
        x_error{alphaInd, 4} = x_error{alphaInd, 4}(1:iterate_rand_fixed)./nTrials;
    end
    
    %%
    
    
end
if lmd >0
    save(sprintf('Regularized_%s_N%d_communication_Density_%f.mat', network_generator,  N, network_Density));
else
    save(sprintf('Unegularized_%s_N%d_communication_Density_%f.mat', network_generator,  N, network_Density));
end




maxCommunicat=5e4;
%% --- Plot figure ---
C = linspecer(8);
methodName = {'Walkman (11b)       '; 'PG-EXTRA          ';  'RW Incremental (decaying stepsize)';'RW Incremental (constant stepsize)'; 'Walkman (11b'')'};
alphaRangeArray = {alphaRangeWalk_ADMM; alphaRange_pgextra;alphaRange_rand; alphaRange_rand_fixed; alphaRangeWalk2};
CommunicateName = {'communication_walk_ADMM',  'communication_pgextra',...
    'communication_rand', 'communication_rand_fixed', 'communication_walkmanb'};
TimeName = {};
LineStyle = {'-', '--', ':', '-.', '-', '--', ':', '-', '-.', '--' ,'-', ':', '-.',....
    '-', '--', ':', '-.', '-', '--', ':', '-', '-.', '--' ,'-', ':', '-.'};
Marker = {'o', '*', '+','s', 'd', 'p' };


fig = figure(10);
plotInd = 0
legendSet = cell(1,1);


TimeName2 = {'time_walk_ADMM',  'time_pgextra', 'time_rand', 'time_rand_fixed', 'time_walkmanb'};

timeLimit2 = min(time_walk_ADMM(end));

plotIncludeZero = 1;
startError = norm(zeros(1,m) - w0', 'fro')^2;
colorInd = [1, 3,  6,5,7,8];
for methodInd = [1,5,2,4,3]
    alphaRange = alphaRangeArray{methodInd};
    for alphaInd =1:max(1,alphaSizeArray(methodInd))
        plotInd = plotInd +1;
        plotNow = max(x_error{alphaInd,methodInd},1e-16);
        communicationRange = eval(sprintf('%s',CommunicateName{methodInd}));
        timeRange2 = eval(sprintf('%s',TimeName2{methodInd}));
        
        if plotIncludeZero
            timeRange2 = [0.001, timeRange2];
            communicationRange = [0.001, communicationRange];
            plotNow = [startError, plotNow];
        end
        
        plotCommuteLength = max(find(communicationRange<=maxCommunicat));
        sub1 = subplot(1,2,1);
        loglog(communicationRange, plotNow,...
            'Color', C(colorInd(methodInd),:),'LineWidth', 3, 'LineStyle', LineStyle{plotInd});
        hold on;
        xlim([1, 1e6]);
        ylim([1e-16,100]);
        sub4 = subplot(1,2,2);
        loglog(timeRange2,plotNow, ...
            'Color', C(colorInd(methodInd),:),'LineWidth', 3, 'LineStyle', LineStyle{plotInd});
        xlim([1,time_pgextra(end)]);
        ylim([1e-16,100]);
        hold on;
        legendSet{plotInd} = [methodName{methodInd}];
    end
end




set(sub1,'YScale', 'log','FontName', 'Times New Roman', 'FontSize', 18);
l1 = legend(sub1,legendSet, 'Location', 'SouthWest');
xlabel(sub1,'Communication Cost','FontName', 'Times New Roman', 'FontSize', 18);
ylabel(sub1, '$\|{\scriptstyle{\mathcal{Y}}}^k - {\scriptstyle{\mathcal{Y}}}^*\|^2/n$','Interpreter', 'latex','FontName', 'Times New Roman', 'FontSize', 18);
set(sub1,'FontName', 'Times New Roman', 'FontSize', 18, 'xtick', [1e1,1e2,1e3,1e4,1e5, 1e6,1e7]);

set(sub4,'FontName', 'Times New Roman', 'FontSize', 18, 'xtick', [1e1,1e2,1e3,1e4, 1e5, 1e6]);
xlabel(sub4,'Running Time','FontName', 'Times New Roman', 'FontSize', 18);





