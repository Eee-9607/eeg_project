clear all;clc;

restoredefaultpath;
addpath(genpath('H:\backup2\5EEGT_day1\toolboxes'));
eeglab;
%close all;
%%transform into fieldtrip
data_root1 = 'H:\erp_new_final\3_3_1';
dirname1 = dir(fullfile(data_root1,'*.set'));
dirname2 = {dirname1.name};

subj_num1 = length(dirname2);
all_EEG_3_3_1 = cell(subj_num1,1);


for i=1:subj_num1
    EEG1= pop_loadset('filename',dirname2{1,i},'filepath',data_root1); 
    all_EEG_3_3_1{i} = EEG1;
end

data_root2 = 'H:\erp_new_final\3_3_2';
dirname3 = dir(fullfile(data_root2,'*.set'));
dirname4 = {dirname3.name};
subj_num2 = length(dirname4);

all_EEG_3_3_2 = cell(subj_num2,1);


for i=1:subj_num2
    EEG2= pop_loadset('filename',dirname4{1,i},'filepath',data_root2); 
    all_EEG_3_3_2{i} = EEG2;
end


% eeglab 2 fieldtrip
restoredefaultpath;
addpath(genpath('H:\backup2\toolboxes\fieldtrip-20180101'));
all_data_tran_3_3_1 = cell(subj_num1,1);
all_data_tran_3_3_2 = cell(subj_num2,1);
for i=1:subj_num1
    all_data_tran_3_3_1{i} = eeglab2fieldtrip(all_EEG_3_3_1{i}, 'preprocessing');
    all_data_tran_3_3_2{i} = eeglab2fieldtrip(all_EEG_3_3_2{i}, 'preprocessing');
end

save('data_trans_3_3.mat','all_data_tran_3_3_1','all_data_tran_3_3_2');


clear all;clc;

%average
 subj_num1=30;

cfg = [];
cfg.channel = 'all';
for i=1:subj_num1
    all_data_tran_3_3_1{i} = ft_timelockanalysis(cfg, all_data_tran_3_3_1{i});
    all_data_tran_3_3_2{i} = ft_timelockanalysis(cfg, all_data_tran_3_3_2{i});
end

merged_1_1_1 = all_data_tran_3_3_1{1};
merged_1_1_2 = all_data_tran_3_3_2{1};
m1_1_1_trials = [];
m1_1_2_trials = [];

 for i=1:subj_num1
    m1_1_1_trials(i,:,:) = all_data_tran_3_3_1{i}.avg;
    m1_1_2_trials(i,:,:) = all_data_tran_3_3_2{i}.avg;
 end    

 
merged_1_1_1.trial = m1_1_2_trials - m1_1_1_trials;
merged_1_1_2.trial = m1_1_2_trials;


cfg         = [];
cfg.channel = {'all'};
cfg.latency = [0 1];

cfg.method           = 'montecarlo';
cfg.statistic        = 'depsamplesT';
cfg.correctm         = 'cluster';


cfg.clusteralpha     = 0.05;
cfg.clusterstatistic = 'maxsum';
cfg.minnbchan        = 2;

cfg_neighb        = [];
cfg_neighb.method = 'distance';
neighbours        = ft_prepare_neighbours(cfg_neighb, merged_1_1_1);

cfg.neighbours    = neighbours;
% same as defined for the between-trials experiment
cfg.tail             = 0;
cfg.clustertail      = 0;
cfg.alpha            = 0.025;
cfg.numrandomization = 5000;

Nsub  = 30;
design = zeros(2, Nsub*2);
design(1,:) = [1:Nsub 1:Nsub];
design(2,:) = [ones(1,Nsub) ones(1,Nsub)*2];

cfg.design = design;
cfg.uvar   = 1;
cfg.ivar   = 2;

[stat1_1] = ft_timelockstatistics(cfg, merged_1_1_1, merged_1_1_2);
save stat1_1;
