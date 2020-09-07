import collections
import time

import numpy as np
import torch

from loader import DataProvider
from model import CrossEntropy, Model
from test import test

batch_size = 64
test_size = 1024
epochs = 10

dest = 'model_q8.pth'


def amino_count(t):
    c = collections.Counter(t)
    keys, values = c.keys(), c.values()
    return list(keys), list(values)


def acid_accuracy(out, target, seq_len):
    out = out.cpu().data.numpy()
    target = target.cpu().data.numpy()
    seq_len = seq_len.cpu().data.numpy()

    out = out.argmax(axis=2)

    count_1 = np.zeros(3)
    count_2 = np.zeros(3)
    for o, t, l in zip(out, target, seq_len):
        o, t = o[:l], t[:l]

        keys, values = amino_count(t)
        count_1[keys] += values

        keys, values = amino_count(t[np.equal(o, t)])
        count_2[keys] += values

    return np.divide(count_2, count_1, out=np.zeros(8), where=count_1!=0)


def timestamp():
    return time.strftime("%Y%m%d%H%M", time.localtime())


def show_progress(e, e_total, train_loss, test_loss, acc):
    print(f'[{e:3d}/{e_total:3d}] train_loss:{train_loss:.2f}, test_loss:{test_loss:.2f}, acc:{acc:.3f}')


def train(model, device, train_loader, optimizer, loss_function):
    model.train()
    train_loss = 0
    length = len(train_loader)
    print('Training...')
    for batch_idx, (data, target, seq_len) in enumerate(train_loader):
        data, target, seq_len = data.to(device), target.to(device), seq_len.to(device)
        optimizer.zero_grad()
        data = data.float()
        out = model(data)
        loss = loss_function(out, target, seq_len)
        loss.backward()
        optimizer.step()
        print('Batch', batch_idx, 'loss', loss.item())
        train_loss += loss.item()

    train_loss /= length

    return train_loss


def main():
    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    data = DataProvider(batch_size, test_size)
    train_loader, test_loader = data.get_data()

    model = Model().to(device)
    loss_function = CrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)

    for e in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, loss_function)
        test_loss, acc = test(model, device, test_loader, loss_function)
        show_progress(e+1, epochs, train_loss, test_loss, acc)

    torch.save(model.state_dict(), dest)


if __name__ == '__main__':
    main()
