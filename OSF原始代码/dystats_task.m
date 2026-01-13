% data direction
data_dir = '';
out_dir = '';

load([data_dir '\Trady_results_mtd.mat'])  % the output of dynamic analysis
load([data_dir '\Multi_AUT.mat'])   
load([data_dir '\meanRTrunx.mat'])

% conditions
trifiles = cell(31,4);

subjects = unique(Multi_AUT.Var1, 'stable');
for s = 1:length(subjects)
    sbjbeha = Multi_AUT(find(strcmp(Multi_AUT{:,1}, subjects{s,1})==1),:);
for r  = 1:3
   sbjcons = sbjbeha(find(sbjbeha{:,2} == r),5).cue;
   trifiles{s,r} = sbjcons';
end
end

% abstract data
for s = [1:3 5:length(trifiles)]
% caculate switch times
substats = WTBT{s,3};
dystats = zeros(length(substats),2);
dystats(1,2) = 3;
for i = 1:length(substats)-1
    dystats(i+1,2) = substats(i+1,1)+substats(i,1);
end
dystats(:,1) = substats;

% caculate the switch time between aut and oct conditions
subtri = trifiles{s,3}; % run1/2/3
st = 0;
for i = 1:length(subtri)
     if strcmp(subtri{1,i},'au')==1
         ntrial = meanRT(s,i); dntrial = 71-5-ntrial;
         dystats(st+1:st+71,3) = [ones(1,5)*11 ones(1,ntrial) ones(1,dntrial)*12];  % 1 presents aut
         st = st+71;
         elseif strcmp(subtri{1,i},'oc')==1
         ntrial = meanRT(s,i); dntrial = 41-5-ntrial;
         dystats(st+1:st+41,3) = [ones(1,5)*21 ones(1,ntrial)*2 ones(1,dntrial)*22];   % 1 presents oct
         st = st+41;
     end
     dystats = dystats(1:545,:);
end

% merge switch times, switch ratio, state1, state2, in aut and oct
dyresults(s,1) = length(find(dystats(find(dystats(:,3)==1),2)==3));        %aut
dyresults(s,5) = length(find(dystats(find(dystats(:,3)==2),2)==3));        %oct
dyresults(s,2) = length(find(dystats(find(dystats(:,3)==1),2)==3))/length(find(dystats(:,3)==1)); %aut
dyresults(s,6) = length(find(dystats(find(dystats(:,3)==2),2)==3))/length(find(dystats(:,3)==2)); %oct
dyresults(s,3) = length(find(dystats(find(dystats(:,3)==1),1)==1));        %aut
dyresults(s,4) = length(find(dystats(find(dystats(:,3)==1),1)==2));        %aut
dyresults(s,7) = length(find(dystats(find(dystats(:,3)==2),1)==1));        %oct
dyresults(s,8) = length(find(dystats(find(dystats(:,3)==2),1)==2));        %oct
end

