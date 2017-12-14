import torch
import os
from model import DNN, CNN
from torch.autograd import Variable
from torch import optim, nn
from torch.utils.data import DataLoader
from data_feeder import ASVDataSet, load_data
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm


# parameters
print_str = "*"*20 + "{}" + "*"*20
batch_size = 256
num_epochs = 10
feature_type = "cqcc"
mode = "train"
save_dir = "./result/cnn/"
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


def get_test_accuracy(data_loader, net):
    correct = 0
    total = 0
    net.eval()
    for i, tmp in enumerate(data_loader):
        data = Variable(tmp['data'])
        label = tmp['label'].long().view(-1)
        if use_cuda():
            data = data.cuda()
        predict = net(data)
        _, predict_label = torch.max(predict.data, 1)
        correct += (predict_label.cpu() == label).sum()
        total += label.size(0)
    acc = correct / total
    return acc


def main():
    # do thing before training
    prepare()
    # loading train data
    print(print_str.format("Begin to loading Data"))
    if mode == "train":
        train_data, train_label = load_data("train", train_protocol, mode=mode, feature_type=feature_type)
        train_dataset = ASVDataSet(train_data, train_label, mode=mode)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

        dev_data, dev_label = load_data("dev", dev_protocol, mode=mode, feature_type=feature_type)
        dev_dataset = ASVDataSet(dev_data, dev_label, mode=mode)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    elif mode == "final":
        train_data, train_label = load_data(["train", "dev"], final_protocol, mode=mode, feature_type=feature_type)
        train_dataset = ASVDataSet(train_data, train_label, mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    print(print_str.format("Done!"))

    model = CNN(45, 2, 0.5)

    if use_cuda():
        model = model.cuda()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.004, momentum=0.9, weight_decay=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[3, 7], gamma=0.1)

    best_dev_accuracy = 0
    best_train_accuracy = 0
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        train_loss = 0
        model.train()
        scheduler.step()
        for tmp in tqdm(train_dataloader, desc="Epoch {}".format(epoch + 1)):
            data = Variable(tmp['data'])
            label = Variable(tmp['label']).view(-1)
            if use_cuda():
                data, label = data.cuda(), label.cuda()

            optimizer.zero_grad()
            predict = model(data)

            _, predict_label = torch.max(predict.data, 1)
            correct += (predict_label.cpu() == label.cpu().data).sum()
            total += label.size(0)

            loss = cross_entropy(predict, label.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]

        train_accuracy = correct / total
        if mode == "final":
            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy
                save_checkpoint(
                    {'state_dict': model.cpu(), 'epoch': epoch + 1, 'acc': train_accuracy},
                    is_best=True,
                    filename="best_eval.pkl"
                )
            save_checkpoint(
                {'state_dict': model.cpu(), 'epoch': epoch + 1, 'acc': train_accuracy},
                is_best=True,
                filename="final_eval.pkl"
            )
            if use_cuda():
                model.cuda()

        if mode == "train":
            dev_accuracy = get_test_accuracy(dev_dataloader, model)

            save_checkpoint(
                {'state_dict': model.cpu(), 'epoch': epoch + 1, 'acc': dev_accuracy},
                is_best=False,
                filename='final_dev.pkl'
            )

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                save_checkpoint(
                    {'state_dict': model.cpu(), 'epoch': epoch + 1, 'acc': dev_accuracy},
                    is_best=True,
                    filename="best_dev.pkl"
                )

            if use_cuda():
                model.cuda()

            tqdm.write("Epoch [%d/%d], Loss: %.4fe-4,  Train Acc %.2f%% Dev Acc %.2f%%" % (
                epoch + 1, num_epochs, 1e4 * train_loss / total, train_accuracy * 100, dev_accuracy * 100
            ))
            print(print_str.format("Best Acc: {}".format(best_dev_accuracy)))


if __name__ == '__main__':
    main()
