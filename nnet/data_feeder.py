import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.io as scio
import extract_feature

rnn_max_frames = 700


def remove_zeros_frames(x, eps=1e-7):
    T, D = x.shape
    s = np.sum(np.abs(x), axis=1)
    s[s < eps] = 0.
    return x[s > eps]


def load_label(label_file):
    labels = {}
    wav_lists = []
    encode = {'spoof': 0, 'genuine': 1}
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_lists.append(line[0])
                wav_id = line[0].replace(".wav", "")
                tmp_label = encode[line[1]]
                labels[wav_id] = tmp_label
    return labels, wav_lists


def load_generael_label():
    labels = {}
    wav_lists = {}
    with open("../data/protocol/ASVspoof2017_train.trn.txt"):
        pass


# this will load data by frame
def load_data(dataset, label_file, mode="train", feature_type="mfcc"):
    extend_num = 6
    mat_pattern = "../../../data/asvspoof/{}_{}.mat"
    if mode != "final":
        labels, wav_lists = load_label(label_file)
    else:
        labels = {}
        wav_lists = []
        train_labels, train_wav_lists = load_label(label_file[0])
        dev_labels, dev_wav_lists = load_label(label_file[1])
        labels.update(train_labels)
        labels.update(dev_labels)
        wav_lists.extend(train_wav_lists)
        wav_lists.extend(dev_wav_lists)
        dataset = "final"

    if mode == "train" or mode == "final":
        final_data = []
        final_label = []

        for wav_name in tqdm(wav_lists, desc="load {} data".format(dataset)):
            wav_id = wav_name.replace(".wav", "")
            label = labels[wav_id]
            if "T" in wav_id:
                wav_path = "../data/ASVspoof2017_train/{}.wav".format(wav_id)
            if "D" in wav_id:
                wav_path = "../data/ASVspoof2017_dev/{}.wav".format(wav_id)

            if feature_type != 'cqcc':
                feature = extract_feature.extract(wav_path, feature_type)
            else:
                mat_path = mat_pattern.format(wav_id, feature_type)
                tmp_data = scio.loadmat(mat_path)
                feature = tmp_data['x']

            # if feature_type == 'fft':
                # extend_num = 2
            feature = np.pad(feature, [[0, 0], [extend_num - 1, extend_num - 1]], mode="edge")
            for i in range(extend_num-1, feature.shape[1] - extend_num):
                tmp_feature = feature[:, i-extend_num+1:i+extend_num].reshape(-1)
                final_data.append(tmp_feature)
                final_label.append(label)
        return final_data, final_label

    elif mode == "test":
        final_data = []
        final_label = []
        final_wav_ids = []

        for wav_name in tqdm(wav_lists, desc="load {} data".format(dataset)):
            wav_id = wav_name.replace(".wav", "")
            label = labels[wav_id]
            if "E" in wav_id:
                wav_path = "../data/ASVspoof2017_eval/{}.wav".format(wav_id)
            if "D" in wav_id:
                wav_path = "../data/ASVspoof2017_dev/{}.wav".format(wav_id)
            if feature_type != 'cqcc':
                feature = extract_feature.extract(wav_path, feature_type)
            else:
                mat_path = mat_pattern.format(wav_id, feature_type)
                tmp_data = scio.loadmat(mat_path)
                feature = tmp_data['x']

            feature = np.pad(feature, [[0, 0], [extend_num - 1, extend_num - 1]], mode="edge")

            final_feature = []
            for i in range(extend_num-1, feature.shape[1] - extend_num):
                tmp_feature = feature[:, i - extend_num + 1:i + extend_num].reshape(-1)
                final_feature.append(tmp_feature)
            final_feature = np.array(final_feature).astype(np.float32)
            final_data.append(final_feature)
            final_label.append(label)
            final_wav_ids.append(wav_id)
        return final_data, final_label, final_wav_ids

    else:
        raise ValueError("the mode doesn't exist")


class ASVDataSet(Dataset):

    def __init__(self, data, label, wav_ids=None, transform=True, mode="train", lengths=None):
        super(ASVDataSet, self).__init__()
        self.data = data
        self.label = label
        self.wav_ids = wav_ids
        self.transform = transform
        self.lengths = lengths
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.lengths is None:
            if self.mode == "train":
                each_data, each_label = self.data[idx], self.label[idx]
            else:
                each_data, each_label, each_wav_id = self.data[idx], self.label[idx], self.wav_ids[idx]
            if self.transform:
                each_data, each_label = torch.from_numpy(each_data).float(), torch.LongTensor([each_label])
            return {
                "data": each_data,
                "label": each_label
            } if self.mode == "train" else {
                "data": each_data,
                "label": each_label,
                "wav_id": each_wav_id
            }
        else:
            each_data, each_label, each_wav_id, each_length = self.data[idx], self.label[idx], \
                                                              self.wav_ids[idx], self.lengths[idx]
            if self.transform:
                each_data, each_label = torch.from_numpy(each_data).float(), torch.LongTensor([each_label])
            return {
                "data": each_data, "label": each_label, "wav_id": each_wav_id, "length": each_length
            }


def load_rnn_data(dataset, label_file, mode="train", feature_type="mfcc"):
    mat_pattern = "../../../data/asvspoof/{}_{}.mat"
    if mode != "final":
        labels, wav_lists = load_label(label_file)
    else:
        labels = {}
        wav_lists = []
        train_labels, train_wav_lists = load_label(label_file[0])
        dev_labels, dev_wav_lists = load_label(label_file[1])
        labels.update(train_labels)
        labels.update(dev_labels)
        wav_lists.extend(train_wav_lists)
        wav_lists.extend(dev_wav_lists)
        dataset = "final"

    final_data = []
    final_label = []
    final_wav_ids = []
    final_lengths = []

    for wav_name in tqdm(wav_lists[0:100], desc="load {} data".format(dataset)):
        wav_id = wav_name.replace(".wav", "")
        label = labels[wav_id]
        if "E" in wav_id:
            wav_path = "../data/ASVspoof2017_eval/{}.wav".format(wav_id)
        if "D" in wav_id:
            wav_path = "../data/ASVspoof2017_dev/{}.wav".format(wav_id)
        if "T" in wav_id:
            wav_path = "../data/ASVspoof2017_train/{}.wav".format(wav_id)

        if feature_type != 'cqcc':
            feature = extract_feature.extract(wav_path, feature_type)

            if feature_type == "mfcc" or feature_type == "imfcc":
                feature_delta = librosa.feature.delta(feature)
                feature_delta_delta = librosa.feature.delta(feature_delta)

                feature = np.concatenate((feature, feature_delta, feature_delta_delta), axis=0)
        else:
            mat_path = mat_pattern.format(wav_id, feature_type)
            tmp_data = scio.loadmat(mat_path)
            feature = tmp_data['x']

        feature = feature.T
        final_lengths.append(feature.shape[0])

        feature = np.pad(feature, [[0, rnn_max_frames-feature.shape[0]], [0, 0]], mode="constant")

        final_data.append(feature)
        final_label.append(label)
        final_wav_ids.append(wav_id)

    return final_data, final_label, final_wav_ids, final_lengths


# this will load data by wav
def load_cnn_data(dataset, label_file, mode="train", feature_type="mfcc"):
    mat_pattern = "../../../data/asvspoof/{}_{}.mat"
    if mode != "final":
        labels, wav_lists = load_label(label_file)
    else:
        labels = {}
        wav_lists = []
        train_labels, train_wav_lists = load_label(label_file[0])
        dev_labels, dev_wav_lists = load_label(label_file[1])
        labels.update(train_labels)
        labels.update(dev_labels)
        wav_lists.extend(train_wav_lists)
        wav_lists.extend(dev_wav_lists)
        dataset = "final"

    final_data = []
    final_label = []
    final_wav_ids = []

    for wav_name in tqdm(wav_lists, desc="load {} data".format(dataset)):
        wav_id = wav_name.replace(".wav", "")
        label = labels[wav_id]
        if "T" in wav_id:
            wav_path = "../data/ASVspoof2017_train/{}.wav".format(wav_id)
        if "D" in wav_id:
            wav_path = "../data/ASVspoof2017_dev/{}.wav".format(wav_id)
        if "E" in wav_id:
            wav_path = "../data/ASVspoof2017_eval/{}.wav".format(wav_id)

        if feature_type != 'cqcc':
            feature = extract_feature.extract(wav_path, feature_type)
        else:
            mat_path = mat_pattern.format(wav_id, feature_type)
            tmp_data = scio.loadmat(mat_path)
            feature = tmp_data['x']

        if feature_type == 'cqcc':
            if feature.shape[1] < 300:
                feature = np.pad(feature, [[0, 0], [0, 300 - feature.shape[1]]], mode='constant')
            feature = feature[:, 0:300]
            final_data.append(feature.T)
        elif feature_type == "raw":
            if feature.shape[0] < 50000:
                feature = np.pad(feature, [0, 50000-feature.shape[0]], mode='constant')
            feature = feature[0:50000]
            final_data.append(feature)
        else:
            if feature.shape[1] < 300:
                feature = np.pad(feature, [[0, 0], [0, 300 - feature.shape[1]]], mode='constant')
            feature = feature[0:700, 0:300]
            final_data.append(feature.T)

        final_label.append(label)
        final_wav_ids.append(wav_id)

    final_data = np.array(final_data)
    if mode != "test":
        return final_data, final_label
    else:
        return final_data, final_label, final_wav_ids











