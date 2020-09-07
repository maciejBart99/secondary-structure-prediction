import torch
from torch import nn
import numpy as np

from softmax_patch import softmax

conv_hidden_size = 128
rnn_size = 128
ffnn_hidden_size = 126
num_classes = 8

# W = [0.48573635202284904,
#      9.106638739641198,
#      0.4302426138099274,
#      2.3732111920639816,
#      483.09178743961354,
#      0.269774468544297,
#      1.128833801799361,
#      0.8249463784853984]

W = [1, 1, 1, 1, 1, 1, 1, 1]


def accuracy(out, target, seq_len):
    out = out.cpu().data.numpy()
    target = target.cpu().data.numpy()
    seq_len = seq_len.cpu().data.numpy()

    out = out.argmax(axis=2)
    return np.array([np.equal(o[:l], t[:l]).sum() / l
                     for o, t, l in zip(out, target, seq_len)]).mean()


class CrossEntropy:

    def __call__(self, out, target, seq_len):
        loss = 0
        for o, t, l in zip(out, target, seq_len):
            t = t.long()
            loss += nn.CrossEntropyLoss()(o[:l], t[:l])
        return loss


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(42 + 2, conv_hidden_size, 11, 1, 11 // 2, padding_mode='reflect'),
            nn.ReLU(),
            nn.BatchNorm1d(conv_hidden_size),
            nn.Dropout(0.3))

        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_hidden_size, conv_hidden_size, 5, 1, 5 // 2, padding_mode='reflect'),
            nn.ReLU(),
            nn.BatchNorm1d(conv_hidden_size),
            nn.Dropout(0.2))

        self.conv3 = nn.Sequential(
            nn.Conv1d(conv_hidden_size, 64, 3, 1, 3 // 2, padding_mode='reflect'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2))

        self.fc = nn.Sequential(
            nn.Linear(64, ffnn_hidden_size),
            # nn.ReLU(),
            # nn.Linear(ffnn_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(ffnn_hidden_size, num_classes))

    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out = out_3.transpose(1, 2)
        out = self.fc(out)
        out = softmax(out, torch.FloatTensor(W), dim=2)
        return out
