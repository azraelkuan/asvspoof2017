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


% read train protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

fmax = 8000;
fmin = 7000;
B=2048;
d=2048;
cf=29;

frame_length = 0.025; %20ms
frame_hop = 0.01; %10ms
n_MFCC = 19; %number of cepstral coefficients excluding 0'th coefficient [default 19]
fl=0.375;
fh=0.5;

%%%%% CQCC
% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));

%% Feature extraction for training data

% extract features for GENUINE training data and store in cell array
disp('Extracting CQCC features for GENUINE training data...');
cqcc_genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_train',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    tmp_fea = cqcc(x, fs, B, fmax, fmin, d, cf, 'ZsdD');
    cqcc_genuineFeatureCell{i} = tmp_fea
end
disp('Done!');

disp('Extracting MFCC features for GENUINE training data...');
mfcc_genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_train',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    tmp_fea = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, fl, fh)';
    mfcc_genuineFeatureCell{i} = tmp_fea
end
disp('Done!');

% extract features for SPOOF training data and store in cell array
disp('Extracting CQCC features for SPOOF training data...');
cqcc_spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_train',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    tmp_fea = cqcc(x, fs, B, fmax, fmin, d, cf, 'ZsdD');
    cqcc_spoofFeatureCell{i} = tmp_fea
end
disp('Done!');

disp('Extracting MFCC features for SPOOF training data...');
mfcc_spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_train',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    tmp_fea = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, fl, fh)';
    mfcc_spoofFeatureCell{i} = tmp_fea
end
disp('Done!');


%% GMM training

% train GMM for GENUINE data
disp('Training CQCC GMM for GENUINE...');
[cqcc_genuineGMM.m, cqcc_genuineGMM.s, cqcc_genuineGMM.w] = vl_gmm([cqcc_genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');

disp('Training MFCC GMM for GENUINE...');
[mfcc_genuineGMM.m, mfcc_genuineGMM.s, mfcc_genuineGMM.w] = vl_gmm([mfcc_genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');

% train GMM for SPOOF data
disp('Training CQCC GMM for SPOOF...');
[cqcc_spoofGMM.m, cqcc_spoofGMM.s, cqcc_spoofGMM.w] = vl_gmm([cqcc_spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');

disp('Training MFCC GMM for SPOOF...');
[mfcc_spoofGMM.m, mfcc_spoofGMM.s, mfcc_spoofGMM.w] = vl_gmm([mfcc_spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
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
cqcc_scores = zeros(size(filelist));
mfcc_scores = zeros(size(filelist));
disp('Computing scores for development trials...');
parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_dev',filelist{i});
    [x,fs] = audioread(filePath);
    % featrue extraction
    x_cqcc = cqcc(x, fs, B, fmax, fmin, d, cf, 'ZsdD');
    x_mfcc = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, fl, fh)';

    %score computation
    llk_genuine1 = mean(compute_llk(x_cqcc,cqcc_genuineGMM.m,cqcc_genuineGMM.s,cqcc_genuineGMM.w));
    llk_spoof1 = mean(compute_llk(x_cqcc,cqcc_spoofGMM.m,cqcc_spoofGMM.s,cqcc_spoofGMM.w));

    llk_genuine2 = mean(compute_llk(x_mfcc,mfcc_genuineGMM.m,mfcc_genuineGMM.s,mfcc_genuineGMM.w));
    llk_spoof2 = mean(compute_llk(x_mfcc,mfcc_spoofGMM.m,mfcc_spoofGMM.s,mfcc_spoofGMM.w));
    % compute log-likelihood ratio
    scores(i) = llk_genuine1 + llk_genuine2 - llk_spoof1 - llk_spoof2;
    cqcc_scores(i) = llk_genuine1 - llk_spoof1
    mfcc_scores(i) = llk_genuine2 - llk_spoof2
end
disp('Done!');

% compute dev performance
[Pmiss,Pfa] = rocch(cqcc_scores(strcmp(labels,'genuine')),cqcc_scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('cqcc EER is %.2f\n', EER);

% compute dev performance
[Pmiss,Pfa] = rocch(mfcc_scores(strcmp(labels,'genuine')),mfcc_scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('mfcc EER is %.2f\n', EER);

% compute dev performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('combine EER is %.2f\n', EER);

% read eval protocol
fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% process each development trial: feature extraction and scoring
scores = zeros(size(filelist));
cqcc_scores = zeros(size(filelist));
mfcc_scores = zeros(size(filelist));
disp('Computing scores for eval trials...');
parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_eval',filelist{i});
    [x,fs] = audioread(filePath);
    % featrue extraction
    x_cqcc = cqcc(x, fs, B, fmax, fmin, d, cf, 'ZsdD');
    x_mfcc = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, fl, fh)';

    %score computation
    llk_genuine1 = mean(compute_llk(x_cqcc,cqcc_genuineGMM.m,cqcc_genuineGMM.s,cqcc_genuineGMM.w));
    llk_spoof1 = mean(compute_llk(x_cqcc,cqcc_spoofGMM.m,cqcc_spoofGMM.s,cqcc_spoofGMM.w));

    llk_genuine2 = mean(compute_llk(x_mfcc,mfcc_genuineGMM.m,mfcc_genuineGMM.s,mfcc_genuineGMM.w));
    llk_spoof2 = mean(compute_llk(x_mfcc,mfcc_spoofGMM.m,mfcc_spoofGMM.s,mfcc_spoofGMM.w));
    % compute log-likelihood ratio
    scores(i) = llk_genuine1 + llk_genuine2 - llk_spoof1 - llk_spoof2;
    cqcc_scores(i) = llk_genuine1 - llk_spoof1
    mfcc_scores(i) = llk_genuine2 - llk_spoof2
end

disp('Done!');

% compute dev performance
[Pmiss,Pfa] = rocch(cqcc_scores(strcmp(labels,'genuine')),cqcc_scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('cqcc EER is %.2f\n', EER);

% compute dev performance
[Pmiss,Pfa] = rocch(mfcc_scores(strcmp(labels,'genuine')),mfcc_scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('mfcc EER is %.2f\n', EER);

% compute dev performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('combine EER is %.2f\n', EER);

function parsave(fname, x)
    save(fname, 'x', '-v6')
end
