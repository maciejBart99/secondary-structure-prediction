import torch
from torch import nn

conv_hidden_size = 64
rnn_size = 128
ffnn_hidden_size = 126
num_classes = 8


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(44, 44, 1, 1, 1 // 2, padding_mode='reflect'),
            nn.ReLU(),
            nn.BatchNorm1d(44),
            nn.Dropout(0.1))

        self.conv1 = nn.Sequential(
            nn.Conv1d(conv_hidden_size, 64, 11, 1, 11 // 2, padding_mode='reflect'),
            nn.ReLU(),
            nn.BatchNorm1d(conv_hidden_size),
            nn.Dropout(0.1))

        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_hidden_size, conv_hidden_size, 5, 1, 5 // 2, padding_mode='reflect'),
            nn.ReLU(),
            nn.BatchNorm1d(conv_hidden_size),
            nn.Dropout(0.1))

        self.conv3 = nn.Sequential(
            nn.Conv1d(44, conv_hidden_size, 3, 1, 3 // 2, padding_mode='reflect'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1))
        self.fc = nn.Sequential(
            nn.Linear(64, ffnn_hidden_size),
            # nn.ReLU(),
            # nn.Linear(ffnn_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(ffnn_hidden_size, num_classes))

    def forward(self, x):
        out_0 = self.conv0(x)
        out_1 = self.conv3(out_0)
        out_2 = self.conv2(out_1)
        out_3 = self.conv1(out_2)
        out = out_3.transpose(1, 2)
        out = self.fc(out)
        out = torch.nn.functional.softmax(out, dim=2)
        return out
