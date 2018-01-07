%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASVspoof 2017 CHALLENGE:
% Audio replay detection challenge for automatic speaker verification anti-spoofing
% 
% http://www.spoofingchallenge.org/
% 
% ====================================================================================
% Matlab implementation of the baseline system for replay detection based
% on constant Q cepstral coefficients (CQCC) features + Gaussian Mixture Models (GMMs)
% ====================================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('voicebox'));

% set paths to the wave files and protocols
%pathToDatabase = fullfile('..','ASVspoof2017_train_dev','wav');
pathToDatabase = '/mnt/speechlab/users/hedi7/data/ASVspoof2017/'
trainProtocolFile = fullfile(pathToDatabase, 'protocol', 'ASVspoof2017_train.trn.txt');
devProtocolFile = fullfile(pathToDatabase, 'protocol', 'ASVspoof2017_dev.trl.txt');
evaProtocolFile = fullfile(pathToDatabase, 'protocol', 'ASVspoof2017_eval_v2_key.trl.txt');

frame_length = 0.025; %20ms
frame_hop = 0.01; %10ms
n_MFCC = 19; %number of cepstral coefficients excluding 0'th coefficient [default 19]
fl=0.002;
fh=0.125;

% read train protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));

%% Feature extraction for training data

% extract features for GENUINE training data and store in cell array
disp('Extracting features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    save_name = strrep(filelist{genuineIdx(i)}, '.wav', '_mfcc.mat');
    save_path = strcat('./features/', save_name);
    filePath = fullfile(pathToDatabase,'ASVspoof2017_train',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    tmp_fea = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, fl, fh)';
    genuineFeatureCell{i} = tmp_fea
    % parsave(save_path, tmp_fea)
end
disp('Done!');

% extract features for SPOOF training data and store in cell array
disp('Extracting features for SPOOF training data...');
spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    save_name = strrep(filelist{spoofIdx(i)}, '.wav', '_mfcc.mat');
    save_path = strcat('./features/', save_name);
    filePath = fullfile(pathToDatabase,'ASVspoof2017_train',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    tmp_fea = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, fl, fh)';
    spoofFeatureCell{i} = tmp_fea
    % parsave(save_path, tmp_fea)
end
disp('Done!');

%% GMM training

% train GMM for GENUINE data
disp('Training GMM for GENUINE...');
[genuineGMM.m, genuineGMM.s, genuineGMM.w] = vl_gmm([genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');

% train GMM for SPOOF data
disp('Training GMM for SPOOF...');
[spoofGMM.m, spoofGMM.s, spoofGMM.w] = vl_gmm([spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');


%% Feature extraction and scoring of development data

% read development protocol
fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% process each development trial: feature extraction and scoring
scores = zeros(size(filelist));
disp('Computing scores for development trials...');
parfor i=1:length(filelist)
    save_name = strrep(filelist{i}, '.wav', '_mfcc.mat');
    save_path = strcat('./features/', save_name);

    filePath = fullfile(pathToDatabase,'ASVspoof2017_dev',filelist{i});
    [x,fs] = audioread(filePath);
    % featrue extraction
    tmp_fea = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, fl, fh)';

    x_mfcc = tmp_fea
    % parsave(save_path, tmp_fea)

    %score computation
    llk_genuine = mean(compute_llk(x_mfcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(x_mfcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
    % compute log-likelihood ratio
    scores(i) = llk_genuine - llk_spoof;
end
disp('Done!');

% compute dev performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);

% read eval protocol
fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% process each development trial: feature extraction and scoring
scores = zeros(size(filelist));
disp('Computing scores for eval trials...');
parfor i=1:length(filelist)
    save_name = strrep(filelist{i}, '.wav', '_mfcc.mat');
    save_path = strcat('./features/', save_name);

    filePath = fullfile(pathToDatabase,'ASVspoof2017_eval',filelist{i});
    [x,fs] = audioread(filePath);
    % featrue extraction
    tmp_fea = melcepst(x,fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, fl, fh)';
    x_mfcc = tmp_fea
    % parsave(save_path, tmp_fea)

    %score computation
    llk_genuine = mean(compute_llk(x_mfcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(x_mfcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
    % compute log-likelihood ratio
    scores(i) = llk_genuine - llk_spoof;
end


disp('Done!');

% compute dev performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('EER is %.2f\n', EER);

function parsave(fname, x)
    save(fname, 'x', '-v6')
end
