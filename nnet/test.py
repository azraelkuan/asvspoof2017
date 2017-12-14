import torch
from model import DNN, CNN
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
from data_feeder import ASVDataSet, load_data


def get_args():
    parser = argparse.ArgumentParser(description="generate score")
    parser.add_argument('--pkl', type=str, default=None, help="the pkl path")
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
    print("model acc: {}".format(pkl['acc']))

    if use_cuda:
        net = net.cuda()

    if args.dt == "eval":
        protocol = "../data/protocol/ASVspoof2017_eval_v2_key.trl.txt"
    else:
        protocol = "../data/protocol/ASVspoof2017_dev.trl.txt"

    test_data, test_label, test_wav_ids = load_data(args.dt, protocol, mode="test", feature_type=args.ft)
    test_dataset = ASVDataSet(test_data, test_label, wav_ids=test_wav_ids, mode="test")

    scores = {}
    correct = 0
    total = 0
    net.eval()
    for i, tmp in enumerate(test_dataset):
        data = Variable(tmp['data'])
        label = tmp['label']
        wav_id = tmp['wav_id']

        if use_cuda:
            data = data.cuda()
        predict = net(data)
        _, predict_label = torch.max(predict.data, 1)
        label = label.repeat(data.size(0), )
        correct += (predict_label.cpu() == label).sum()
        total += data.size(0)
        scores[wav_id] = torch.sum(predict_label) / predict_label.size(0)
    print("final acc: {}".format(correct / total))
    with open('{}_score.txt'.format(args.dt), 'w', encoding='utf-8') as f:
        for k, v in scores.items():
            f.write("{} {}\n".format(k, v))


if __name__ == '__main__':
    main()