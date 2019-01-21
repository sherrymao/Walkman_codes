clear;clc;close all;
addpath ../functions/
addpath ./functions_least_square/


seed=[2017];
fprintf('Seed = %d\n',seed);
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

maxCommunicat=1e5;
N = 50; m = 3;
maxIteration =1e3;

networkRadius = 15;
lmd = 0;
U=randn(N,m);
x_init = randn(m, 1);
sigma =1e-2;
d = U*x_init +sigma*randn(N, 1);
x0 = (inv((U'*U)./N + lmd.*eye(m))*(U'*d))./N;
x0_m=repmat(x0',N,1);
v0_m=df(U, d, x0_m, N, m, lmd);


testing_par = 0;

alphaRangeEXTRA = [0.1];%complete: 0.11;ring: 0.08;linear: 0.02; radius: 0.13; %opt for N =20:2
alphaRangeDiffusion = [0.08];%[0.2];%complete: 0.22;ring: 0.08;linear: 0.02;%radius: 0.13; [3:2:10];
alphaRangewadmm = [2.5];%[2.4];%(50,20):[3];%(100,15):[5];%[0.045];
alphaRangeADMM = [0.15];
alphaRangerand = [0];%[1e-3, 1e-2];
alphaRangerand_fixed = [1e-4];%[1e-4];%[1e-4];
alphaRangeWalk2 = [5e-2]%[1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2];

alphaSizeArray = [length(alphaRangeEXTRA), length(alphaRangeDiffusion)...
    ,  length(alphaRangewadmm), length(alphaRangeADMM), length(alphaRangerand_fixed), length(alphaRangerand),...
    length(alphaRangeWalk2)];
alphaSize = max(alphaSizeArray);
x_error = cell(alphaSize, 7);

%% --- Network Gen ---
network_generator = 'radius'%'complete'
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
        [cmat,incimat,nnum, Coordinates]=NetworkGen(N,30,30,networkRadius);
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

%% --- Aextra Simulation ---

time_mu = 1;

for alphaInd = 1:alphaSize
    
    flagAlpha_extra = alphaInd <= alphaSizeArray(1);
    flagAlpha_diffusion = alphaInd <= alphaSizeArray(2);
    flagAlpha_walkmana = alphaInd <= alphaSizeArray(3);
    flagAlpha_ADMM = alphaInd <= alphaSizeArray(4);
    flagAlpha_rand_fixed = alphaInd <= alphaSizeArray(5);
    flagAlpha_rand = alphaInd <= alphaSizeArray(6);
    flagAlpha_walkmanb = alphaInd <= alphaSizeArray(7);
    
    
    if flagAlpha_walkmanb
        %% preparation for the application of Walkman(b)
        dataInd_walkmanb = 1;
        alpha_walkmanb = alphaRangeWalk2(alphaInd)
        z_tilde_walkmanb = zeros(1, m);
        
        
        %% initialization for Walkman(b)
        z_walkmanb = zeros(N, m);
        y_walkmanb = zeros(N, m);
        gramInv = cell(N, 1);
        for nodeInd = 1:N
            gramInv{nodeInd} = pinv(U(nodeInd, :)'*U(nodeInd, :)+ alpha_walkmanb*eye(m));
        end
        xbar_walkmanb = zeros(1, m);
        communicateRound_walkmanb = 0;
        iteration_walkmanb = 0;
        time_walkmanb = zeros(1, maxCommunicat);
        for testind = 1:N
            g_0_walkmanb(testind, :) = (U(testind,:)'*(U(testind,:)* x0 -d(testind)))' + lmd.* x0';
        end
        while communicateRound_walkmanb < maxCommunicat || iteration_walkmanb < maxIteration
            iteration_walkmanb = iteration_walkmanb + 1;
            time_walkmanb(iteration_walkmanb) = exprnd(time_mu, 1);
            proposeInd = Neighbor_noloop{dataInd_walkmanb}(randperm(degreeVec_noloop(dataInd_walkmanb), 1));
            randvar = rand(1);
            dataInd_walkmanb = proposeInd;
            g_now_walkmanb = (U(dataInd_walkmanb,:)'*(U(dataInd_walkmanb,:)*y_walkmanb(dataInd_walkmanb, :)'-d(dataInd_walkmanb)))' + lmd.* y_walkmanb(dataInd_walkmanb, :);
            
            ypre_walkmanb = y_walkmanb(dataInd_walkmanb, :);
            zpre_walkmanb = z_walkmanb(dataInd_walkmanb, :);
            y_walkmanb(dataInd_walkmanb, :) =  xbar_walkmanb + z_walkmanb(dataInd_walkmanb, :) - alpha_walkmanb.*g_now_walkmanb;
            z_walkmanb(dataInd_walkmanb, :) = xbar_walkmanb - y_walkmanb(dataInd_walkmanb, :) + z_walkmanb(dataInd_walkmanb, :);
            xbar_walkmanb = xbar_walkmanb + ((y_walkmanb(dataInd_walkmanb, :) - z_walkmanb(dataInd_walkmanb, :))- (ypre_walkmanb - zpre_walkmanb))./N;
            x_error{alphaInd, 7}(iteration_walkmanb)=  norm(y_walkmanb - x0_m, 'fro')^2/N;
            
            communicateRound_walkmanb = communicateRound_walkmanb + 1;
            
        end
        communication_walkmanb = [1:iteration_walkmanb];
        time_walkmanb = cumsum(time_walkmanb(1:iteration_walkmanb));
        x_error{alphaInd, 7} = x_error{alphaInd, 7}(1:iteration_walkmanb);
    end
    
    
    
    if flagAlpha_rand_fixed
        x_error{alphaInd,5} = zeros(1, 1e6);
        alpha_rand_fixed = alphaRangerand_fixed(alphaInd)
        time_rand_fixed = zeros(1, 1e6);
        communicateRound_rand_fixed= 0;
        iterate_rand_fixed = 0;
        dataInd_rand_fixed = randperm(N,1);
        x_rand_fixed = zeros(m, 1);
        while communicateRound_rand_fixed < 1e6
            iterate_rand_fixed = iterate_rand_fixed + 1;
            time_rand_fixed(iterate_rand_fixed) = time_rand_fixed(iterate_rand_fixed) + exprnd(time_mu);
            proposeInd = Neighbor_noloop{dataInd_rand_fixed}(randperm(degreeVec_noloop(dataInd_rand_fixed),1));
            randvar = rand(1);
            dataInd_rand_fixed = proposeInd;
            x_rand_fixed = x_rand_fixed - alpha_rand_fixed.*((U(dataInd_rand_fixed,:)'*(U(dataInd_rand_fixed,:)*x_rand_fixed - d(dataInd_rand_fixed))) + lmd.* x_rand_fixed);
            x_error{alphaInd,5}(iterate_rand_fixed) = x_error{alphaInd,5}(iterate_rand_fixed) + norm(x_rand_fixed-x0,'fro')^2;
            communicateRound_rand_fixed = communicateRound_rand_fixed + 1;
        end
        communication_rand_fixed = [1:iterate_rand_fixed];
        time_rand_fixed = cumsum(time_rand_fixed(1:iterate_rand_fixed));
        x_error{alphaInd, 5} = x_error{alphaInd, 5}(1:iterate_rand_fixed);
    end
    
    if flagAlpha_rand
        alpha_rand = alphaRangerand(alphaInd);
        x_error{alphaInd,6} = zeros(1, 1e6);
        time_rand = zeros(1,1e6);
        dataInd_rand = randperm(N,1);
        communicateRound_rand = 0;
        iterate_rand = 0;
        x_rand = zeros(m, 1);
        while communicateRound_rand < 1e6
            iterate_rand = iterate_rand + 1;
            activateTime = exprnd(time_mu, [1, degreeVec_noloop(dataInd_rand)]);
            [time_temp, propose_neighbor] = min(activateTime);
            time_rand(iterate_rand) = time_rand(iterate_rand) + exprnd(time_mu);
            proposeInd = Neighbor_noloop{dataInd_rand}(randperm(degreeVec_noloop(dataInd_rand),1));
            randvar = rand(1);
            dataInd_rand = proposeInd;
            x_rand = x_rand - ((U(dataInd_rand,:)'*(U(dataInd_rand,:)*x_rand - d(dataInd_rand))) + lmd.* x_rand).*min(1e-2, 5/(iterate_rand));
            x_error{alphaInd,6}(iterate_rand) =  x_error{alphaInd,6}(iterate_rand) + norm(x_rand-x0,'fro')^2;
            communicateRound_rand = communicateRound_rand + 1;
        end
        communication_rand = [1:iterate_rand];
        time_rand = cumsum(time_rand(1:iterate_rand));
        x_error{alphaInd, 6} = x_error{alphaInd, 6}(1:iterate_rand);
    end
    
    
    if flagAlpha_ADMM
        alpha_ADMM = alphaRangeADMM(alphaInd)
        gramUInv =  cell(N, 1);
        gramU = cell(N, 1);
        for nodeInd = 1:N
            gramU{nodeInd} = U(nodeInd, :)'* U(nodeInd, :) + (2*alpha_ADMM*degreeVec_noloop(nodeInd)).*eye(m);
            gramUInv{nodeInd} = inv(gramU{nodeInd});
        end
        x_ADMM = zeros(N, m);
        a_ADMM = zeros(N, m);
        communicateRound_ADMM = 0;
        iteration_ADMM = 0;
        time_ADMM = zeros(1, maxCommunicat);
        while communicateRound_ADMM < maxCommunicat || iteration_ADMM < maxIteration
            iteration_ADMM = iteration_ADMM + 1;
            temp_ADMM = alpha_ADMM.*(D_noloop + A_noloop)*x_ADMM - a_ADMM;
            x_ADMM = df_conjugate(U, gramUInv, d, temp_ADMM, N, m);
            a_ADMM = a_ADMM  + alpha_ADMM.*(D_noloop*x_ADMM - A_noloop*x_ADMM);
            communicateRound_ADMM = communicateRound_ADMM + 2*N_eg;
            x_error{alphaInd,4}(iteration_ADMM) = norm(x_ADMM-x0_m,'fro')^2./N;
            time_inner = exprnd(time_mu, [2*N_eg,1]);
            maxtime1 = max(time_inner);
            time_ADMM(iteration_ADMM) = maxtime1;
        end
        communication_ADMM = [1:iteration_ADMM].*(2*N_eg);
        time_ADMM = cumsum(time_ADMM(1:iteration_ADMM));
        x_error{alphaInd, 4} = x_error{alphaInd, 4}(1:iteration_ADMM);
    end
    
    
    
    
    if flagAlpha_walkmana
        %% preparation for the application of Walkman(a)
        dataInd_walkmana = 1;
        alpha_walkmana = alphaRangewadmm(alphaInd)
        z_tilde_walkmana = zeros(1, m);
        flag_sppg = 1;
        %% criterion
        z0_m = x0_m - alpha_walkmana.*v0_m;
        %% initialization for Walkman(a)
        z_walkmana = zeros(N, m);
        y_walkmana = zeros(N, m);
        gramInv = cell(N, 1);
        for nodeInd = 1:N
            gramInv{nodeInd} = pinv(U(nodeInd, :)'*U(nodeInd, :)+ alpha_walkmana*eye(m));
        end
        xbar_walkmana = zeros(1, m);
        communicateRound_walkmana = 0;
        iteration_walkmana = 0;
        time_walkmana = zeros(1, maxCommunicat);
        for testind = 1:N
            g_0_walkmana(testind, :) = (U(testind,:)'*(U(testind,:)* x0 -d(testind)))' + lmd.* x0';
        end
        while communicateRound_walkmana < maxCommunicat || iteration_walkmana < maxIteration
            iteration_walkmana = iteration_walkmana + 1;
            time_walkmana(iteration_walkmana) = exprnd(time_mu, 1);
            proposeInd = Neighbor_noloop{dataInd_walkmana}(randperm(degreeVec_noloop(dataInd_walkmana), 1));
            randvar = rand(1);
            dataInd_walkmana = proposeInd;
            ypre_walkmana = y_walkmana(dataInd_walkmana, :);
            zpre_walkmana = z_walkmana(dataInd_walkmana, :);
            y_walkmana(dataInd_walkmana, :) =  (gramInv{dataInd_walkmana}*(U(dataInd_walkmana, :)'*d(dataInd_walkmana) + alpha_walkmana.*(xbar_walkmana+z_walkmana(dataInd_walkmana, :))'))';
            z_walkmana(dataInd_walkmana, :) = xbar_walkmana - y_walkmana(dataInd_walkmana, :) + z_walkmana(dataInd_walkmana, :);
            xbar_walkmana = xbar_walkmana + ((y_walkmana(dataInd_walkmana, :) - z_walkmana(dataInd_walkmana, :))- (ypre_walkmana - zpre_walkmana))./N;
            x_error{alphaInd, 3}(iteration_walkmana)=  norm(y_walkmana - x0_m, 'fro')^2/N;
            communicateRound_walkmana = communicateRound_walkmana + 1;
        end
        communication_walkmana = [1:iteration_walkmana];
        time_walkmana = cumsum(time_walkmana(1:iteration_walkmana));
        x_error{alphaInd, 3} = x_error{alphaInd, 3}(1:iteration_walkmana);
    end
    
    
    if flagAlpha_extra
        %% EXTRA
        alpha_extra = alphaRangeEXTRA(alphaInd)
        x_extra = zeros(N, m);
        g_extra = df(U, d, x_extra, N, m, lmd);
        x_extra_p1 = W*x_extra - alpha_extra.*g_extra;
        communicateRound_extra = 0;
        iteration_extra = 0;
        time_extra = zeros(1, maxCommunicat);
        while communicateRound_extra < maxCommunicat || iteration_extra < maxIteration
            g_extra_p1 = df(U, d, x_extra_p1, N, m, lmd);
            g_extra = df(U, d, x_extra, N, m, lmd);
            x_extra_p2 = Wtilde*(2.*x_extra_p1-x_extra) - alpha_extra.*g_extra_p1+ alpha_extra.*g_extra;
            x_extra = x_extra_p1;
            x_extra_p1 = x_extra_p2;
            
            communicateRound_extra = communicateRound_extra + 2*N_eg;
            iteration_extra = iteration_extra + 1;
            x_error{alphaInd,1}(iteration_extra) = norm(x_extra_p1-x0_m,'fro')^2./N;
            time_inner = exprnd(time_mu, [2*N_eg,1]);
            maxtime1 = max(time_inner);
            time_extra(iteration_extra) = maxtime1;
        end
        communication_extra = [1:iteration_extra].*(2*N_eg);
        time_extra = cumsum(time_extra(1:iteration_extra));
        x_error{alphaInd, 1} = x_error{alphaInd, 1}(1:iteration_extra);
    end
    
    
    
    if flagAlpha_diffusion
        %% exact diffusion
        alpha_diffusion = alphaRangeDiffusion(alphaInd)
        x_diffusion = zeros(N,m);
        g_diffusion = df(U, d, x_diffusion, N, m, lmd);
        psi_diffusion = x_diffusion - alpha_diffusion.*g_diffusion;
        x_diffusion_p1 = Wtilde * psi_diffusion;
        idx=0;
        communicateRound_diffusion = 0;
        iteration_diffusion = 0;
        time_diffusion = zeros(1, maxCommunicat);
        while communicateRound_diffusion < maxCommunicat || iteration_diffusion < maxIteration
            g_diffusion_p1 = df(U, d, x_diffusion_p1, N, m, lmd);
            psi_diffusion_p1 = x_diffusion_p1 - alpha_diffusion.*g_diffusion_p1;
            phi_diffusion_p1 = x_diffusion_p1 + psi_diffusion_p1 - psi_diffusion;
            x_diffusion_p2 = Wtilde * phi_diffusion_p1;
            psi_diffusion = psi_diffusion_p1;
            x_diffusion_p1 = x_diffusion_p2;
            communicateRound_diffusion = communicateRound_diffusion + 2*N_eg;
            iteration_diffusion = iteration_diffusion + 1;
            x_error{alphaInd,2}(iteration_diffusion) = norm(x_diffusion_p1-x0_m,'fro')^2./N;
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



%% plot 2*2 subplot, communication, iteration; time in model 1, time in model 2
C = linspecer(8);
fig = figure(9);
plotInd = 0
legendSet = cell(1,1);
TimeName2 = {'time_extra', 'time_diffusion', 'time_walkmana','time_ADMM', 'time_rand_fixed', 'time_rand', 'time_walkmanb'};

timeLimit2 = min(time_walkmanb(end));

colorInd = [4,2,1,3,5,6,7];
plotIncludeZero = 1;
startError = norm(zeros(N,m) - x0_m, 'fro')^2./N;

for methodInd =[3,7,4,1,2,5,6]
    alphaRange = alphaRangeArray{methodInd};
    for alphaInd =1:max(1,alphaSizeArray(methodInd))
        plotInd = plotInd +1;
        plotNow = max((x_error{alphaInd,methodInd}),1e-16);
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
            'Color', C(colorInd(methodInd),:),'LineWidth', 3, 'LineStyle', LineStyle{plotInd});%colorInd(methodInd)
        hold on;
        xlim([1, 1e6]);
        ylim([1e-16, 1]);
        sub4 = subplot(1,2,2);
        plotCommuteLength = max(find(timeRange2<=timeLimit2));
        loglog(timeRange2,plotNow, ...
            'Color', C(colorInd(methodInd),:),'LineWidth', 3, 'LineStyle', LineStyle{plotInd});
        xlim([1, 1e6]);
        ylim([1e-16, 1]);
        hold on;
        legendSet{plotInd} = [methodName{methodInd}];%, 'alpha=', sprintf('%.5f', alphaRangeArray{methodInd}(alphaInd))];%];%
    end
    
end
set(sub1,'YScale', 'log','FontName', 'Times New Roman', 'FontSize', 18);
l1 = legend(sub1,legendSet, 'Location', 'SouthWest');
xlabel(sub1,'Communication Cost','FontName', 'Times New Roman', 'FontSize', 18);

ylabel(sub1, '$\|{\scriptstyle{\bf\mathcal{Y}}}^k - {\scriptstyle{\bf\mathcal{Y}}}^*\|^2/n$','Interpreter', 'latex','FontName', 'Times New Roman', 'FontSize', 18);
set(sub1,'FontName', 'Times New Roman', 'FontSize', 18, 'xtick', [1e1,1e2,1e3,1e4,1e5, 1e6]);
set(sub4,'FontName', 'Times New Roman', 'FontSize', 18, 'xtick', [1e1,1e2,1e3,1e4,1e5,1e6]);
xlabel(sub4,'Running Time','FontName', 'Times New Roman', 'FontSize', 18);

