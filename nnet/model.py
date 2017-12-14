import torch
from torch import nn
import torch.nn.init as init


class DNN(nn.ModuleList):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNN, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),

                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
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


class CNN(nn.Module):

    def __init__(self, input_dim, output_dim, drop_out):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_out = drop_out

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.linear = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(drop_out),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(drop_out),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        batch_size, dim = x.size()
        x = x.view(batch_size, 1, self.input_dim, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x