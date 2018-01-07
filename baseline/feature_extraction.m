clear; close all; clc;

addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('voicebox'));


pathToDatabase = fullfile('/speechlab/users/hedi7/data/ASVspoof2017/');

trainProtocolFile = fullfile(pathToDatabase, 'protocol', 'ASVspoof2017_train.trn.txt');
devProtocolFile = fullfile(pathToDatabase, 'protocol', 'ASVspoof2017_dev.trl.txt');
evaProtocolFile = fullfile(pathToDatabase, 'protocol', 'ASVspoof2017_eval_v2_key.trl.txt');


frame_length = 0.02; %20ms
frame_hop = 0.01; %10ms
n_MFCC = 13; %number of cepstral coefficients excluding 0'th coefficient [default 19]
delta_feature = '0'; % 0'th coefficient. Append any of the following for more options 
                     %'d': for single delta; 'D': for double delta; 'E': log energy

%{% read train protocol%}
%fileID = fopen(trainProtocolFile);
%protocol = textscan(fileID, '%s%s%s%s%s%s%s');
%fclose(fileID);

%% get file and label lists
%filelist = protocol{1};
%labels = protocol{2};

%% get indices of genuine and spoof files
%genuineIdx = find(strcmp(labels,'genuine'));
%spoofIdx = find(strcmp(labels,'spoof'));


%disp('Extracting features for GENUINE training data...');
%genuineCqccTrain = cell(size(genuineIdx));
%genuineMfccTrain = cell(size(genuineIdx));

%parfor i=1:length(genuineIdx)
    %filePath = fullfile(pathToDatabase,'ASVspoof2017_train',filelist{spoofIdx(i)});
    %[x,fs] = audioread(filePath);
    %genuineCqccTrain{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    %genuineMfccTrain{i} = melcepst(x,fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs);
%end
%disp('Done!');

%save ('./features/genuineCqccTrain.mat', 'genuineCqccTrain');
%save ('./features/genuineMfccTrain.mat', 'genuineMfccTrain');

%disp('Extracting features for SPOOF training data...');
%spoofCqccTrain = cell(size(spoofIdx));
%spoofMfccTrain = cell(size(spoofIdx));

%parfor i=1:length(spoofIdx)
    %filePath = fullfile(pathToDatabase,'ASVspoof2017_train',filelist{spoofIdx(i)});
    %[x,fs] = audioread(filePath);
    %spoofCqccTrain{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    %spoofMfccTrain{i} = melcepst(x,fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs);
%end
%disp('Done!');

%save ('./features/spoofCqccTrain.mat', 'spoofCqccTrain');
%save ('./features/spoofMfccTrain.mat', 'spoofMfccTrain');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEVELOPMENT SET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% read train protocol
%fileID = fopen(devProtocolFile);
%protocol = textscan(fileID, '%s%s%s%s%s%s%s');
%fclose(fileID);

%% get file and label lists
%filelist = protocol{1};
%labels = protocol{2};

%% get indices of genuine and spoof files
%genuineIdx = find(strcmp(labels,'genuine'));
%spoofIdx = find(strcmp(labels,'spoof'));


%disp('Extracting features for GENUINE development data...');
%genuineCqccDev = cell(size(genuineIdx));
%genuineMfccDev = cell(size(genuineIdx));

%parfor i=1:length(genuineIdx)
     %filePath = fullfile(pathToDatabase,'ASVspoof2017_dev',filelist{spoofIdx(i)});
    %[x,fs] = audioread(filePath);
    %genuineCqccDev{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    %genuineMfccDev{i} = melcepst(x,fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs);
%end

%disp('Done!');

%save ('./features/genuineCqccDev.mat', 'genuineCqccDev');
%save ('./features/genuineMfccDev.mat', 'genuineMfccDev');

%disp('Extracting features for SPOOF development data...');
%spoofCqccDev = cell(size(spoofIdx));
%spoofMfccDev = cell(size(spoofIdx));

%parfor i=1:length(spoofIdx)
     %filePath = fullfile(pathToDatabase,'ASVspoof2017_dev',filelist{spoofIdx(i)});
    %[x,fs] = audioread(filePath);
    %spoofCqccDev{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    %spoofMfccDev{i} = melcepst(x,fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs);
%end
%disp('Done!');

%save ('./features/spoofCqccDev.mat', 'spoofCqccDev');
%{s%}ave ('./features/spoofMfccDev.mat', 'spoofMfccDev');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EVALUATION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read train protocol
fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};
display(filelist)

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));


disp('Extracting features for evaluation data...');
genuineCqccEva = cell(size(genuineIdx));
genuineMfccEva = cell(size(genuineIdx));

parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_eval',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    evaluationCqcc{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    evaluationMFCC{i} = melcepst(x,fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs);
end
disp('Done!');

save ('./features/genuineCqccEva.mat', 'genuineCqccEva');
save ('./features/genuineMfccEva.mat', 'genuineMfccEva');

disp('Extracting features for SPOOF Evalution data...');
spoofCqccEva = cell(size(spoofIdx));
spoofMfccEva = cell(size(spoofIdx));

parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_eval',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    spoofCqccDev{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    spoofMfccDev{i} = melcepst(x,fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs);
end
disp('Done!');

save ('./features/spoofCqccEva.mat', 'spoofCqccEva');
save ('./features/spoofMfccEva.mat', 'spoofMfccEva');


