%------------ Email by chenqunlin@swu.edu.cn ------------------------------
clc; clear

%------------ Step 1: Set-Up ----------------------------------------------
data_dir = 'D:\DATA\CCP\DyfcTask\data';
data_dir1 = 'D:\DATA\CCP\DyfcTask\analysis';
out_dir = 'D:\DATA\CCP\DyfcTask\analysis\output';
load([data_dir '\rd_out.mat']);

%------------- Altas and ROIs ---------------------------------------------
dncn = [89:145,245:293];   

%------------- the number of run in the task-fmri data analysis------------
run = 1;

%------------- Step 2: fliter subjects ------------------------------------
for s = 1:length(rd_out)
subID = rd_out{s,4}
ts_data = rd_out{s,run};

% delete Low SNR nodes 
denodes = [];
nodes = setdiff(dncn,denodes);
ts_data = ts_data(:,nodes);

% check data
ts_data(:,find(sum(ts_data)==0))=[];
ts_data(:,max(ts_data)==min(ts_data))=[];

%------------- Step 3:  Time-resolved Functional Connectivity -------------
% ref. Shine et al., 2015; https://github.com/macshine/coupling/
window = 14; 
% forward facing windows
direction = 1;                                   
% delete zeros from the end of the 3d matrix 
trim = 1;                                       

% MTD; Shine et al., 2015
mtd = coupling(ts_data,window,direction,trim);

%------------- Step 4: Graph Theoretical Measures -------------------------
nNodes = size(mtd,1); nTime = size(mtd,3);

% Dynamic Modularity method #1:
ci = zeros(nNodes,nTime);
q = zeros(nTime,1);

for t = 1:nTime
    for loop = 1:100
     % this function should be run multiple times in order to obtain consensus
        [cc(:,loop),qq(loop,1)] = modularity_louvain_und_sign(mtd(:,:,t));  
    end
        ci(:,t) = consensus_und(agreement_weighted(cc,qq),0.1,500);
        q(t,1) = mean(qq);
end

% Module Degree Z-score (WT)
WT = zeros(nNodes,nTime);

for t = 1:nTime
  WT(:,t) = module_degree_zscore(mtd(:,:,t),ci(:,t),0);
end
WTBT{s,1} = WT;

%Participation index (BT)
BT = zeros(nNodes,nTime);
for t = 1:nTime
  BT(:,t) = participation_coef_sign(mtd(:,:,t),ci(:,t));
end
WTBT{s,2} = BT;

%------------- Step 5: 2-dimensional Cartographic Profile (CP)-------------
% generate a 100 x 100 2d histogram
xbins = [0:0.01:1.0]; ybins = [5 : -.1 : -5];    
CP = zeros(size(xbins,2),size(ybins,2),nTime);
xNumBins = numel(xbins); yNumBins = numel(ybins);

for t = 1:nTime
  Xi = round(interp1(xbins, 1:xNumBins, BT(:,t), 'linear', 'extrap') );
  Yi = round(interp1(ybins, 1:yNumBins, WT(:,t), 'linear', 'extrap') );
  Xi = max( min(Xi,xNumBins), 1);
  Yi = max( min(Yi,yNumBins), 1);
  CP(:,:,t) = accumarray([Yi(:) Xi(:)],1,[yNumBins xNumBins],@sum);
end

%------------- Step 6: Plot figure ---------------------------------------- 
figure(1)
set (gcf,'position',[0 0 1500 1000]);
subplot(3,3,1);
plot(mean(BT))
ylabel('Mean Between-Module degree')
xlabel('Time windows')
title('Dynamic Fluctuations in Cartography')
subplot(3,3,2);
plot(mean(WT))
ylabel('Mean Within-Module degree')
xlabel('Time windows')
title('Dynamic Fluctuations in Cartography')

% wd & pc
xticklabels = xbins;
xticks = linspace(1, 101, numel(xticklabels)); 
yticklabels = ybins;
yticks = linspace(1, 101, numel(yticklabels));
subplot(1,3,3);
imagesc(squeeze(mean(CP,3)))
imagesc(mean(CP,3))
set(gca,  'XTickLabel', xticklabels)
set(gca,  'YTickLabel',{'5','4','3','2','1','0','-1','-2','-3','-4','-5'})
%xlim([0 1]);ylim([-5 5])
ylabel('Module degree')
xlabel('Participation Coefficient')
title('Segregated and Integrated States')
caxis([0 1])
hcb = colorbar
title(hcb,'Percent Time')
set(gcf,'color','w')
print(gcf,'-dpng',[out_dir '\InSe_' subID '_dy#1.png']); 

%------------- setp 7: clustering -----------------------------------------
pcwd = reshape(CP,xNumBins * yNumBins,nTime);
[idx] = kmeans(pcwd',2,'Distance','sqeuclidean','Replicates',100);

state1=find(idx==1);
state2=find(idx==2);

% t-test to define states
s1_WT = mean(WT(:,state1)')';
s2_WT = mean(WT(:,state2)')';
[H,P,CI,STATS] = ttest2(s1_WT,s2_WT);

s1_BT = mean(BT(:,state1)')';
s2_BT = mean(BT(:,state2)')';
[H,P,CI,STATS] = ttest2(s1_BT,s2_BT);

BT_value = [];
if STATS.tstat >0 
   Trady_results(s,1) = size(state1,1);Trady_results(s,2) = size(state2,1);
   BT_value(1,1:2) = [mean(s1_BT) mean(s2_BT)];
   WTBT{s,3}=idx;
else
   Trady_results(s,1) = size(state2,1);Trady_results(s,2) = size(state1,1);
   BT_value(1,1:2) = [mean(s2_BT) mean(s1_BT)];
   WTBT{s,3}=3-idx;
end

for c = 1:length(idx)-1
    switch_f(c,1) = idx(c,1)+idx(c+1,1);
end
Trady_results(s,3) = size(find(switch_f==3),1);

%------------- step 8: figure the two states ------------------------------
% figure states patter
CP_state1 = CP(:,:,state1);
CP_state2 = CP(:,:,state2);

% wd & pc
%figure(2)
xticklabels = xbins;
xticks = linspace(1, 101, numel(xticklabels)); 
yticklabels = ybins;
yticks = linspace(1, 101, numel(yticklabels));

%state1
subplot(3,3,4);  
imagesc(squeeze(mean(CP_state1,3)))
set(gca, 'XTickLabel', xticklabels)
set(gca, 'YTickLabel',{'5','4','3','2','1','0','-1','-2','-3','-4','-5'})
% xlim([0 1]);ylim([-5 5])
ylabel('Module degree')
xlabel('Participation Coefficient')
title('States #1')
caxis([0 1])
hcb = colorbar
title(hcb,'Percent Time')
set(gcf,'color','w')

%state2
subplot(3,3,5);  
imagesc(squeeze(mean(CP_state2,3)))
set(gca, 'XTickLabel', xticklabels)
set(gca, 'YTickLabel',{'5','4','3','2','1','0','-1','-2','-3','-4','-5'})
% xlim([0 1]);ylim([-5 5])
ylabel('Module degree')
xlabel('Participation Coefficient')
title('States #2')
caxis([0 1])
hcb = colorbar
title(hcb,'Percent Time')
set(gcf,'color','w')

% participations coefficient
subplot(3,3,7); 
bar(BT_value);
ylim([0 1]);
set(gca, 'XTickLabel', {'Cooperation(S1)','Antagonism(S2)'})
ylabel('Mean PC of states')
text(1.2,0.9,['T-value = ', num2str(STATS.tstat) '; P-value = ' num2str(P)])
set(gcf,'color','w')
% switch patterns
subplot(3,3,8); 
plot(idx,'k-o','linewidth',1.5 ,'markersize',1);
ylim([1 3]);
ylabel('States')
xlabel('Time windows')
title('States Switch Pattern')
set(gcf,'color','w')
print(gcf,'-dpng',[out_dir '\InSe_' subID '_dy#1_' num2str(run) '.png']);
end
%------------- Step 9: save the results -----------------------------------
save([out_dir '\Trady_results_mtd'],'Trady_results');
save([out_dir '\WTBT'],'WTBT');