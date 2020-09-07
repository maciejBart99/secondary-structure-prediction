import torch
from torch import nn
import numpy as np

from softmax_patch import softmax

conv_hidden_size = 24
ffnn_hidden_size = 126
num_classes = 8
#
# W = [4.85736352e-06, 9.10663874e-05, 4.30242614e-06, 2.37321119e-05,
#      4.83091787e-03, 2.69774469e-06, 1.12883380e-05, 8.24946378e-06]
W = [4.85736352, 91.0663874, 4.30242614, 23.7321119,
     4830.91787, 2.69774469, 11.2883380, 8.24946378]


def accuracy(out, target, seq_len):
    out = out.cpu().data.numpy()
    target = target.cpu().data.numpy()
    seq_len = seq_len.cpu().data.numpy()

    out = out.argmax(axis=2)
    return np.array([np.equal(o[:l], t[:l]).sum() / l
                     for o, t, l in zip(out, target, seq_len)]).mean()


class CrossEntropy:

    def __init__(self):
        pass

    def __call__(self, out, target, seq_len):
        loss = 0
        for o, t, l in zip(out, target, seq_len):
            t = t.long()
            loss += nn.CrossEntropyLoss()(o[:l], t[:l])
        return loss


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.embed = nn.Embedding(20, 5)

        self.conv1 = nn.Sequential(
            nn.Conv1d(22 + 5, conv_hidden_size, 11, 1, 11 // 2),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(22 + 5 + conv_hidden_size, conv_hidden_size, 7, 1, 7 // 2),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv1d(22 + 5 + 2 * conv_hidden_size, conv_hidden_size, 3, 1, 3 // 2),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(conv_hidden_size * 3, ffnn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffnn_hidden_size, num_classes),
            nn.ReLU())

    def forward(self, x):
        x = x.long()
        embed = self.embed(torch.narrow(x, 1, 0, 20).transpose(1, 2).argmax(dim=2)).transpose(1, 2)
        rest = torch.narrow(x, 1, 20, 22)
        x = torch.cat([embed, rest], dim=1)
        out_1 = self.conv1(x)
        out_2 = self.conv2(torch.cat([out_1, x], dim=1))
        conv_out = self.conv3(torch.cat([out_2, out_1, x], dim=1))
        conv_out = torch.cat([out_1, out_2, conv_out], dim=1)
        out = conv_out.transpose(1, 2)
        out = self.fc(out)
        out = softmax(out, torch.FloatTensor(W), dim=2)

        return out
