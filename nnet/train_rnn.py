import torch
import os
from model import RNN
from torch.autograd import Variable
from torch import optim, nn
from torch.utils.data import DataLoader
from data_feeder import load_rnn_data, ASVDataSet
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm


print_str = "*"*20 + "{}" + "*"*20
feature_type = "cqcc"
mode = "train"
num_epochs = 10
batch_size = 4
save_dir = "./result/rnn/"
train_protocol = "../data/protocol/ASVspoof2017_train.trn.txt"
dev_protocol = "../data/protocol/ASVspoof2017_dev.trl.txt"
final_protocol = [train_protocol, dev_protocol]


def prepare():
    # input("*****Please check the save dir --> {} <--, Enter to continue*****".format(save_dir))
    os.system('mkdir -p {}'.format(save_dir))


def use_cuda():
    is_cuda = torch.cuda.is_available()
    return is_cuda


def save_checkpoint(state, is_best=False, filename='final.pkl'):
    torch.save(state, save_dir+filename)
    if is_best:
        torch.save(state, save_dir+"best.pkl")


def main():
    prepare()
    print(print_str.format("Begin to loading Data"))

    net = RNN(90, 256, 2, 2, 0.1)
    if use_cuda():
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    cross_entropy = nn.CrossEntropyLoss()

    if mode == "train":
        train_data, train_label, train_wav_ids, train_lengths = load_rnn_data("train", train_protocol,
                                                                              mode=mode, feature_type=feature_type)
        train_dataset = ASVDataSet(train_data, train_label, wav_ids=train_wav_ids, mode=mode, lengths=train_lengths)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        total_loss = 0
        for tmp in tqdm(train_dataloader, desc="Epoch {}".format(epoch + 1)):
            data = tmp['data']
            label = tmp['label']
            length = tmp['length']

            max_len = int(torch.max(length))
            data = data[:, :max_len, :]
            label = label[:, :max_len]

            sorted_length, indices = torch.sort(
                length.view(-1), dim=0, descending=True
            )
            sorted_length = sorted_length.long().numpy()

            data, label = data[indices], label[indices]

            data, label = Variable(data), Variable(label).view(-1)
            if use_cuda():
                data, label = data.cuda(), label.cuda()

            optimizer.zero_grad()
            outputs, out_length = net(data, sorted_length)
            loss = cross_entropy(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
            _, predict = torch.max(outputs, 1)
            correct += (predict.data == label.data).sum()
            total += label.size(0)

        print("Loss: {} \t Acc: {}".format(total_loss / len(train_dataloader), correct/total))

if __name__ == '__main__':
    main()
