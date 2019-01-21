clear;clc;close all;
addpath ../functions/
addpath ./functions_NNPCA/


seed=[2017];
fprintf('Seed = %d\n',seed);
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

maxCommunicat=1e4;
N=50;m=784;M=m;
maxIteration =1e3;

networkRadius = 15;

load mnist_info_new.mat

alphaRangeWalk_ADMM = 1.5;%[1:0.5:3];%[1.5];%[1.5];%2.5;%3;%[1];%1;%[1, 2, 2.5, 3];%[3];
alphaRange_pgextra = [0.1];%, 5e-2, 0.1, 0.2];%[0.05, 0.1, 0.5, 0.8, 1,2,5];%0.5%, 0.8, 1, 2, 5];%[0.5];%[0.5];%[0.1]
alphaRange_rand = [0];%[0];%[0];
alphaRange_rand_fixed = [1e-3];%[1e-3];%[1e-2];
alphaRangeWalk2 = [1];%[1:0.5:3];%[1]%1:best[10,50,100,200,300,500,1e3];%[1e-3,5e-3,1e-2,2e-2,5e-2];%[0.1, 0.5, 1, 2, 2.5, 3, 5, 10];

cov_mats= cov_mats./1000;
total_cov = total_cov./1000;
alphaSizeArray = [length(alphaRangeWalk_ADMM),length(alphaRange_pgextra),...
    length(alphaRange_rand), length(alphaRange_rand_fixed), length(alphaRangeWalk2)];
alphaSize = max(alphaSizeArray);
func_val = cell(alphaSize, 6);
opt_gap = cell(alphaSize, 6);

con_vio = cell(alphaSize, 6);
lagrang_grad = cell(alphaSize, 6);
time_mu =1;


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
func0 = -w0'*total_cov*w0/2;

%% --- Testing ---

for alphaInd = 1:alphaSize
    
    flagAlpha_walkmana = alphaInd <= alphaSizeArray(1);
    flagAlpha_pgextra = alphaInd <= alphaSizeArray(2);
    flagAlpha_rand = alphaInd <= alphaSizeArray(3);
    flagAlpha_rand_fixed = alphaInd <= alphaSizeArray(4);
    flagAlpha_walkmanb = alphaInd <= alphaSizeArray(5);
    
    
    if flagAlpha_walkmanb
        %% preparation for the application of Walkman(b)
        dataInd_walkmanb = 1;
        alpha_walkmanb = alphaRangeWalk2(alphaInd)
        
        %% initialization for  Walkman(b)
        time_walkmanb = zeros(1, maxCommunicat);
        func_val{alphaInd, 5} = zeros(1, maxCommunicat);
        opt_gap{alphaInd, 5} = zeros(1, maxCommunicat);
        con_vio{alphaInd, 5} = zeros(1, maxCommunicat);
        
        z_walkmanb = zeros(N, m);
        y_walkmanb = ones(N,m)./sqrt(m);
        xbar_walkmanb =ones(1, m)./sqrt(m); 
        communicateRound_walkmanb = 0;
        iteration_walkmanb = 0;
        ave_y = ones(1, m)/sqrt(m);
        
        while communicateRound_walkmanb < maxCommunicat
            iteration_walkmanb = iteration_walkmanb + 1;
            
            time_walkmanb(iteration_walkmanb) = time_walkmanb(iteration_walkmanb)+ exprnd(time_mu);
            proposeInd = Neighbor_noloop{dataInd_walkmanb}(randperm(degreeVec_noloop(dataInd_walkmanb),1));
            dataInd_walkmanb = proposeInd;
            ypre_walkmanb = y_walkmanb(dataInd_walkmanb, :);
            zpre_walkmanb = z_walkmanb(dataInd_walkmanb, :);
            x_walkmanb = prox_nn_norm(xbar_walkmanb);
            y_walkmanb(dataInd_walkmanb, :) = x_walkmanb + z_walkmanb(dataInd_walkmanb, :) + ((cov_mats(:,:, dataInd_walkmanb)*y_walkmanb(dataInd_walkmanb, :)')'./alpha_walkmanb); %(Inv_mats(:,:, dataInd_walkmanb)* (x_walkmanb + z_walkmanb(dataInd_walkmanb, :))')';
            z_walkmanb(dataInd_walkmanb, :) = x_walkmanb - y_walkmanb(dataInd_walkmanb, :) + z_walkmanb(dataInd_walkmanb, :);
            xbar_walkmanb = xbar_walkmanb + ((y_walkmanb(dataInd_walkmanb, :) - z_walkmanb(dataInd_walkmanb, :))- (ypre_walkmanb - zpre_walkmanb))./N;
            
            ave_y = ave_y + (y_walkmanb(dataInd_walkmanb,:) - ypre_walkmanb)./N;
            func_val{alphaInd, 5}(iteration_walkmanb)= -x_walkmanb*total_cov*x_walkmanb'./2 -func0;
            con_vio{alphaInd, 5}(iteration_walkmanb) = norm(y_walkmanb - repmat(x_walkmanb, N, 1), 'fro')^2;
            grad_x = -total_cov*x_walkmanb';
            opt_gap{alphaInd, 5}(iteration_walkmanb) = norm(grad_x' + proj_partialr(-grad_x', x_walkmanb))^2 + con_vio{alphaInd, 5}(iteration_walkmanb);
            communicateRound_walkmanb = communicateRound_walkmanb + 1;
            
        end
        
        communication_walkmanb = [1:iteration_walkmanb];
        time_walkmanb = cumsum(time_walkmanb(1:iteration_walkmanb));
    end
    
    if flagAlpha_walkmana
        %% preparation for the application of  Walkman(a)
        dataInd_walkmana = 1;
        alpha_walkmana = alphaRangeWalk_ADMM(alphaInd)
        Identity = eye(m);
        Inv_mats = zeros(m,m,N);
        for node = [1:N]
            Inv_mats(:, :, node) = inv(Identity - cov_mats(:,:,node)./alpha_walkmana);
        end
        
        %% initialization for  Walkman(a)
        nTrials = 1;
        time_walkmana = zeros(1, maxCommunicat);
        func_val{alphaInd, 1} = zeros(1, maxCommunicat);
        opt_gap{alphaInd, 1} = zeros(1, maxCommunicat);
        con_vio{alphaInd, 1} = zeros(1, maxCommunicat);
        for iTrial = 1:nTrials
            iTrial
            z_walkmana = zeros(N, m);
            y_walkmana = ones(N,m)./sqrt(m);
            xbar_walkmana =ones(1, m)./sqrt(m); 
            communicateRound_walkmana = 0;
            iteration_walkmana = 0;
            
            ave_g = zeros(m, 1);
            for inode = 1:N
                ave_g = ave_g - cov_mats(:,:,inode)*y_walkmana(inode, :)';
            end
            ave_g = ave_g./N;
            ave_y = ones(1, m)/sqrt(m);
            
            while communicateRound_walkmana < maxCommunicat
                iteration_walkmana = iteration_walkmana + 1;
                time_walkmana(iteration_walkmana) = time_walkmana(iteration_walkmana)+ exprnd(time_mu);
                proposeInd = Neighbor_noloop{dataInd_walkmana}(randperm(degreeVec_noloop(dataInd_walkmana),1));
                dataInd_walkmana = proposeInd;
                
                
                ypre_walkmana = y_walkmana(dataInd_walkmana, :);
                zpre_walkmana = z_walkmana(dataInd_walkmana, :);
                x_walkmana = prox_nn_norm(xbar_walkmana);
                y_walkmana(dataInd_walkmana, :) =  (Inv_mats(:,:, dataInd_walkmana)* (x_walkmana + z_walkmana(dataInd_walkmana, :))')';
                z_walkmana(dataInd_walkmana, :) = x_walkmana - y_walkmana(dataInd_walkmana, :) + z_walkmana(dataInd_walkmana, :);
                xbar_walkmana = xbar_walkmana + ((y_walkmana(dataInd_walkmana, :) - z_walkmana(dataInd_walkmana, :))- (ypre_walkmana - zpre_walkmana))./N;
             
                ave_y = ave_y + (y_walkmana(dataInd_walkmana,:) - ypre_walkmana)./N;
                func_val{alphaInd, 1}(iteration_walkmana)= -x_walkmana*total_cov*x_walkmana'./2 -func0;
                con_vio{alphaInd, 1}(iteration_walkmana) = norm(y_walkmana - repmat(x_walkmana, N, 1), 'fro')^2;
                grad_x = -total_cov*x_walkmana';
                opt_gap{alphaInd, 1}(iteration_walkmana) = norm(grad_x' + proj_partialr(-grad_x', x_walkmana))^2 + con_vio{alphaInd, 1}(iteration_walkmana);
                communicateRound_walkmana = communicateRound_walkmana + 1;
                
            end
        end
        communication_walkmana = [1:iteration_walkmana];
        time_walkmana = cumsum(time_walkmana(1:iteration_walkmana)./nTrials);
    end
    
    
    
    %% PG-EXTRA
    if flagAlpha_pgextra
        func_val{alphaInd, 2} = zeros(1, maxIteration);
        opt_gap{alphaInd, 2} = zeros(1, maxIteration);
        con_vio{alphaInd, 2} = zeros(1, maxIteration);
        
        
        alpha_pgextra = alphaRange_pgextra(alphaInd)
        x_pgextra = ones(N, m)./sqrt(m);
        g_pgextra = grad_nnpca(cov_mats, x_pgextra);
        x_pgextra_phalf = W*x_pgextra - alpha_pgextra.*g_pgextra;
        x_pgextra_p1 = prox_nn_norm(x_pgextra_phalf);
        communication_pgextra = [];
        communicateRound_pgextra = 0;
        iteration_pgextra = 0;
        while iteration_pgextra < maxIteration
%             tic
            iteration_pgextra = iteration_pgextra + 1;
            g_pgextra_p1 = grad_nnpca(cov_mats, x_pgextra_p1);
            g_pgextra = grad_nnpca(cov_mats, x_pgextra);
            x_pgextra_p3half = W*x_pgextra_p1 + x_pgextra_phalf - Wtilde*x_pgextra ...
                - alpha_pgextra.*(g_pgextra_p1 - g_pgextra);
            x_pgextra_p2 = prox_nn_norm(x_pgextra_p3half);
            x_pgextra = x_pgextra_p1;
            x_pgextra_phalf = x_pgextra_p3half;
            x_pgextra_p1 = x_pgextra_p2;
            func_val{alphaInd,2}(iteration_pgextra) = norm(x_pgextra_p1-w0_m,'fro')^2./N;
            time_inner = exprnd(time_mu, [2*N_eg,1]);
            maxtime1 = max(time_inner);
            time_pgextra(iteration_pgextra) = maxtime1;
            communicateRound_pgextra = communicateRound_pgextra + 2*N_eg;
            ave_x = mean(x_pgextra, 1);
            ave_g = -(total_cov*ave_x')';
            func_val{alphaInd, 2}(iteration_pgextra)= -ave_x*total_cov*ave_x'./2-func0;
            con_vio{alphaInd, 2}(iteration_pgextra) = norm(x_pgextra - repmat(ave_x, N, 1), 'fro')^2;
            opt_gap{alphaInd, 2}(iteration_pgextra) = norm(ave_g + proj_partialr(-ave_g, ave_x))^2+con_vio{alphaInd, 2}(iteration_pgextra);
 
%             toc
        end
        communication_pgextra = [1:iteration_pgextra].*(2*N_eg);
        time_pgextra = cumsum(time_pgextra(1:iteration_pgextra));
    end
    
    
    %% random walk decaying stepsize
    if flagAlpha_rand
        alpha_rand = alphaRange_rand(alphaInd);
        time_rand = zeros(1, 1e6);
        func_val{alphaInd, 3} = zeros(1, 1e6);
        opt_gap{alphaInd, 3} = zeros(1, 1e6);
        con_vio{alphaInd, 3} = zeros(1, 1e6);
        nTrials = 1;
        for iTrial = 1:nTrials
            dataInd_rand = randperm(N,1);
            communicateRound_rand = 0;
            iterate_rand = 0;
            x_rand =  ones(m, 1)./sqrt(m);
            while communicateRound_rand < 1e6
                iterate_rand = iterate_rand + 1;
                time_rand(iterate_rand) = time_rand(iterate_rand) + exprnd(time_mu);
                proposeInd = Neighbor_noloop{dataInd_rand}(randperm(degreeVec_noloop(dataInd_rand),1));
                randvar = rand(1);
                dataInd_rand = proposeInd;
                grad_now = -cov_mats(:,:,dataInd_rand)*x_rand;
                x_rand = prox_nn_norm(x_rand - grad_now.*min(1e-2, 5/(iterate_rand)));
                func_val{alphaInd,3}(iterate_rand) =  -x_rand' *total_cov *x_rand./2-func0;
                ave_g = -total_cov*x_rand;
                opt_gap{alphaInd, 3}(iterate_rand) = norm(ave_g + proj_partialr(-ave_g, x_rand))^2;
                communicateRound_rand = communicateRound_rand + 1;
                if mod(iterate_rand,1000)==0
                    disp(iterate_rand)
                end
            end
        end
        communication_rand = [1:iterate_rand];
        time_rand = cumsum(time_rand(1:iterate_rand)./nTrials);
    end
    
    
    
    %% random walk fixed stepsize
    if flagAlpha_rand_fixed
        alpha_rand_fixed = alphaRange_rand_fixed(alphaInd)
        time_rand_fixed = zeros(1, 1e6);
        func_val{alphaInd, 4} = zeros(1, 1e6);
        opt_gap{alphaInd, 4} = zeros(1, 1e6);
        con_vio{alphaInd, 4} = zeros(1, 1e6);
        nTrials = 1;
        for iTrial = 1:nTrials
            communicateRound_rand_fixed= 0;
            iterate_rand_fixed = 0;
            dataInd_rand_fixed = randperm(N,1);
            x_rand_fixed =  ones(m, 1)./sqrt(m);
            while communicateRound_rand_fixed < 1e6
                iterate_rand_fixed = iterate_rand_fixed + 1;
                time_rand_fixed(iterate_rand_fixed) = time_rand_fixed(iterate_rand_fixed) + exprnd(time_mu);%time_temp;
                proposeInd = Neighbor_noloop{dataInd_rand_fixed}(randperm(degreeVec_noloop(dataInd_rand_fixed), 1));
                randvar = rand(1);
                dataInd_rand_fixed = proposeInd;
                grad_now = -cov_mats(:,:,dataInd_rand_fixed)*x_rand_fixed;
                x_rand_fixed = prox_nn_norm(x_rand_fixed - alpha_rand_fixed.*grad_now);
                func_val{alphaInd,4}(iterate_rand_fixed) =  -x_rand_fixed' *total_cov *x_rand_fixed./2-func0;
                ave_g = -total_cov*x_rand_fixed;
                opt_gap{alphaInd, 4}(iterate_rand_fixed) = norm(ave_g + proj_partialr(-ave_g, x_rand_fixed))^2;
                communicateRound_rand_fixed = communicateRound_rand_fixed + 1;
%                 if mod(iterate_rand_fixed,1000)==0
%                     disp(iterate_rand_fixed)
%                 end
            end
        end
        communication_rand_fixed = [1:iterate_rand_fixed];
        time_rand_fixed = cumsum(time_rand_fixed(1:iterate_rand_fixed)./nTrials);
    end
    
end




methodName = {'Walkman (11b)        '; 'PG-EXTRA          '; 'RW Incremental (decaying stepsize)';'RW Incremental (constant stepsize)'; 'Walkman (11b'')    '};
alphaRangeArray = {alphaRangeWalk_ADMM; alphaRange_pgextra; alphaRange_rand; alphaRange_rand_fixed; alphaRangeWalk2};
CommunicateName = {'communication_walkmana', 'communication_pgextra',...
    'communication_rand', 'communication_rand_fixed', 'communication_walkmanb'};
LineStyle = {'-', '--', ':', '-.', '-', '--', ':', '-', '-.', '--' ,'-', ':', '-.','-', '--', ':',};
Marker = {'o', '*', '+','s', 'd', 'p' };

C = linspecer(8);
colorInd = [1,3,6,5,7,8];
fig = figure(1);
hold on;
plotInd = 0
legendSet = cell(1,1);


TimeName2 = {'time_walkmana',  'time_pgextra',  'time_rand', 'time_rand_fixed', 'time_walkmanb'};

timeLimit2 = 1e5;

plotIncludeZero = 1;
init_x = ones(1, m)./sqrt(m);
ave_g = -total_cov*init_x';
startopt  = norm(ave_g' + proj_partialr(-ave_g, init_x))^2;
startfunc = -init_x *total_cov *init_x'./2-func0;

for methodInd =[1,5,2,4,3]
    alphaRange = alphaRangeArray{methodInd};
    for alphaInd =1:max(1, alphaSizeArray(methodInd))
        plotInd = plotInd +1;
        plotopt = max(opt_gap{alphaInd,methodInd},1e-16);
        plotfunc = max(func_val{alphaInd,methodInd},1e-16);
        communicationRange = eval(sprintf('%s',CommunicateName{methodInd}));
        timeRange2 = eval(sprintf('%s',TimeName2{methodInd}));
        
        if plotIncludeZero
            timeRange2 = [0.001, timeRange2];
            communicationRange = [0.001, communicationRange];
            plotopt = [startopt, plotopt];
            plotfunc = [startfunc, plotfunc];
        end
        
        plotCommuteLength = max(find(communicationRange<=maxCommunicat));
        sub1 = subplot(2,2,1);
        hold on;
        loglog(communicationRange, plotopt,...
            'Color', C(colorInd(methodInd),:),'LineWidth', 3, 'LineStyle', LineStyle{plotInd});
        
        xlim([1, 1e6]);
        ylim([1e-16,10]);
        box on;
        
        sub4 = subplot(2,2,2);
        hold on;
        loglog(timeRange2,plotopt, ...
            'Color', C(colorInd(methodInd),:),'LineWidth', 3, 'LineStyle', LineStyle{plotInd});
        xlim([1,time_pgextra(end)]);
        ylim([1e-16,10]);
        legendSet{plotInd} = [methodName{methodInd}];
        box on;
        
        sub2 = subplot(2,2,3);
        hold on;
        loglog(communicationRange, plotfunc,...
            'Color', C(colorInd(methodInd),:),'LineWidth', 3, 'LineStyle', LineStyle{plotInd}); 
        xlim([1, 1e6]);
        ylim([1e-16,1]);
        box on;
        
        sub3 = subplot(2,2,4);
        hold on;
        plot(plotfunc(2:end), ...
            'Color', C(colorInd(methodInd),:),'LineWidth', 3, 'LineStyle', LineStyle{plotInd});
        xlim([1,maxIteration]);
        ylim([0.2, 0.203])
        box on;
    end
end

set(sub1,'YScale', 'log','XScale', 'log','FontName', 'Times New Roman', 'FontSize', 18);
xlabel(sub1,'Communication Cost','FontName', 'Times New Roman', 'FontSize', 18);
ylabel(sub1, 'Optimality Gap','FontName', 'Times New Roman', 'FontSize', 18);
set(sub1,'FontName', 'Times New Roman', 'FontSize', 18, 'xtick', [1e1,1e2,1e3,1e4,1e5]);
set(sub4,'YScale', 'log','XScale', 'log','FontName', 'Times New Roman', 'FontSize', 18, 'xtick', [1e1,1e2,1e3,1e4, 1e5, 1e6]);
xlabel(sub4,'Running Time','FontName', 'Times New Roman', 'FontSize', 18);
 l1 = legend(sub4,legendSet, 'Location', 'SouthWest');

set(sub2,'YScale', 'log','XScale', 'log','FontName', 'Times New Roman', 'FontSize', 18);
xlabel(sub2,'Communication Cost','FontName', 'Times New Roman', 'FontSize', 18);
ylabel(sub2, '$f(x^k) - f(x^*)$','Interpreter', 'latex','FontName', 'Times New Roman', 'FontSize', 18);
set(sub2,'FontName', 'Times New Roman', 'FontSize', 18, 'xtick', [1e1,1e2,1e3,1e4,1e5]);
set(sub3,'FontName', 'Times New Roman', 'FontSize', 18, 'xtick',[0.2:0.2:1]*1e3 );
xlabel(sub3,'Iterations','FontName', 'Times New Roman', 'FontSize', 18);
