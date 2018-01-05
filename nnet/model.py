import torch
import math
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


"""
DNN
"""


class DNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNN, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(True),
                nn.Dropout(0.5),

                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(True),
                nn.Dropout(0.5),
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(True),
                nn.Dropout(0.5),
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(True),
                nn.Dropout(0.5),
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(True),
                nn.Dropout(0.5),

                nn.Linear(hidden_dim, output_dim)
                )

        self.init_weight(self.main)

    def forward(self, x):
        out = self.main(x)
        return out

    def init_weight(self, m):
        for each_module in m:
            if "Linear" in each_module.__class__.__name__:
                init.xavier_normal(each_module.weight)
                init.constant(each_module.bias, 0.)


"""
RNN
"""


class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, drop_out):
        super(RNN, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=drop_out, bidirectional=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x, length):

        x = torch.nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x, None)

        out, new_length = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[:, -1, :])
        return out, new_length

    def init_weight(self):
        pass

"""
CNN
"""


class CNN(nn.Module):

    def __init__(self, input_dim, output_dim, drop_out):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_out = drop_out

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 11),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 7),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3)
        )
        self.linear = nn.Sequential(
            nn.Linear(3840, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        batch_size, dim = x.size()
        assert dim == 1001 * 11
        x = x.view(batch_size, 1, self.input_dim, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x


"""
VGG
"""


vgg_cfg = {
    'VGG11': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M']
}


class VGG(nn.Module):

    def __init__(self, input_dim, vgg_name):
        super(VGG, self).__init__()
        self.input_dim = input_dim
        self.features = self.make_layers(vgg_cfg[vgg_name])
        self.linear = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2)
        )
        self._initialize_weights()

    def forward(self, x):
        batch_size, dim = x.size()
        assert dim == 1001 * 11
        x = x.view(batch_size, 1, self.input_dim, -1)

        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(True)
                ]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

"""
Light CNN
"""


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class LCNN(nn.Module):

    def __init__(self, num_classes=2):
        super(LCNN, self).__init__()

        self.features = nn.Sequential(
            mfm(1, 8, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                group(8, 16, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(16, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(32, 24, 3, 1, 1),
            group(24, 24, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.block = nn.Sequential(
            mfm(20064, 256, type=0),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

        self.init_weight()

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        # input()
        out = self.block(x)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight.data)
                m.bias.data.zero_()


# not work, i don't konw now
class RawCNN(nn.Module):

    def __init__(self, input_dim, drop_out):
        super(RawCNN, self).__init__()
        self.input_dim = input_dim
        last_filter_size = 1
        cur_filter_size = 16
        block_list = nn.ModuleList()

        for i in range(8):
            pool_size = 3 if i < 7 else 2
            block_list += self.conv1d(last_filter_size, cur_filter_size, pool_size=pool_size)
            last_filter_size = cur_filter_size
            if drop_out > 0:
                block_list += [nn.Dropout(drop_out)]

            if i < 2:
                cur_filter_size = 16
            elif i < 4:
                cur_filter_size = 64
            else:
                cur_filter_size = 128
        self.convs = nn.Sequential(*block_list)
        self.linear = nn.Linear(1408, 2)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def conv1d(self, in_channels, out_channels, kernel_size=3, pool_size=3):
        return [
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(True),
                nn.MaxPool1d(pool_size)
            )
        ]

