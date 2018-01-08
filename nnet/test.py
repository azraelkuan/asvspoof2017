import torch
import os
from model import DNN, CNN, LCNN
import argparse
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from data_feeder import ASVDataSet, load_data
from tqdm import tqdm
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description="generate score")
    parser.add_argument('--pkl', type=str, default=None, help="the pkl path")
    parser.add_argument('--tm', type=str, default='dnn', help='the training model')
    parser.add_argument('--ft', '--feature_type', type=str, default='cqcc', help="the type of feature")
    parser.add_argument('--dt', '--data_set', type=str, default='dev', help='the dataset u will test')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    use_cuda = True if torch.cuda.is_available() else False

    if args.pkl is None or args.dt is None:
        raise ValueError("some args value must not be None")

    pkl = torch.load(args.pkl)
    net = pkl['state_dict']
    # mean = pkl['mean']
    # std = pkl['std']
    print("model acc: {}".format(pkl['acc']))

    if use_cuda:
        net = net.cuda()

    if args.dt == "eval":
        protocol = "../data/protocol/ASVspoof2017_eval_v2_key.trl.txt"
    else:
        protocol = "../data/protocol/ASVspoof2017_dev.trl.txt"

    test_data, test_label, test_wav_ids = load_data(args.dt, protocol, mode="test", feature_type=args.ft)

    test_dataset = ASVDataSet(test_data, test_label, wav_ids=test_wav_ids, mode="test")
    # test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)

    scores = {}
    net.eval()
    for tmp in tqdm(test_dataset):
        data = Variable(tmp['data'])
        wav_id = tmp['wav_id']

        if use_cuda:
            data = data.cuda()
        predict = net(data)
        predict = F.softmax(predict, dim=1)
        predict = predict.data.cpu().mean(0)
        scores[wav_id] = predict[1]

    save_dir = os.path.join("result", args.tm, args.ft)
    os.system('mkdir -p {}'.format(save_dir))
    with open(os.path.join(save_dir, args.dt+"_score.txt"), 'w', encoding='utf-8') as f:
        for k, v in scores.items():
            f.write("{} {}\n".format(k, v))


if __name__ == '__main__':
    main()
