import torch
import numpy as np

from loader import DataProvider
from model import accuracy, Model, CrossEntropy

batch_size = 128
test_size = 1024

data = DataProvider(batch_size, test_size)
train_loader, test_loader = data.get_data()

print('loaded....')


def test(model, device, test_loader, loss_function, log=False):
    model.eval()
    test_loss = 0
    acc = 0
    res = []
    length = len(test_loader)

    with torch.no_grad():
        for i, (data, target, seq_len) in enumerate(test_loader):
            data, target, seq_len = data.to(device), target.to(device), seq_len.to(device)
            out = model(data)
            test_loss += loss_function(out, target, seq_len).cpu().data.numpy()
            a = accuracy(out, target, seq_len)
            for ind in range(534):
                ac = accuracy(out[ind:ind + 1, :, :], target[ind:ind + 1, :], seq_len[ind:ind + 1])
                seq_str = ''.join([get_amino(x) for x in data[ind, :21, :seq_len[ind]].data.numpy().argmax(axis=0).tolist()])
                pred = ''.join([get_structure_label(x) for x in out[ind, :seq_len[ind], :].data.numpy().argmax(axis=1).tolist()])
                tar = ''.join([get_structure_label(x) for x in target[ind, :seq_len[ind]].data.numpy().tolist()])

                if log:
                    print(ind, 'Accuracy', ac, 'len', int(seq_len[ind]))
                    print(seq_str)
                    print(pred)
                    print(tar)
                    print(''.join(['=' if x else ' ' for x in np.equal(target[ind, :seq_len[ind]].data.numpy(),
                                                                       out[ind, :seq_len[ind], :].data.numpy().argmax(axis=1)).tolist()
                                   ]))
                    print('')
                res.append({"id": ind, "acc": ac, "len": int(seq_len[ind]), 'seq': seq_str, 'pred': pred, 'target': tar,
                            "possibility":  out[ind, :seq_len[ind], :].data.numpy().tolist()})

            acc += a

    test_loss /= length
    acc /= length

    return test_loss, acc, res


def get_amino(num):
    dc = {
        0: 'A', 1: 'C', 2: 'E', 3: 'D', 4: 'G', 5: 'F', 6: 'I', 7: 'H', 8: 'K', 9: 'M', 10: 'L', 11: 'N', 12: 'Q',
        13: 'P', 14: 'S', 15: 'R', 16: 'T', 17: 'W', 18: 'V', 19: 'Y', 20: 'X'
    }

    if num in dc:
        return dc[num]
    else:
        return ' '


def get_structure_label(num):
    dc = {
        0: 'L', 1: 'B', 2: 'E', 3: 'G', 4: 'I', 5: 'H', 6: 'S', 7: 'T',
    }

    if num in dc:
        return dc[num]
    else:
        return ' '


def main(mod, log=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Model().to(device)
    print(mod)
    model.load_state_dict(torch.load(mod))
    loss_function = CrossEntropy()
    test_loss, acc, result_table = test(model, device, test_loader, loss_function, log)
    print('Test loss', test_loss)
    print('Accuracy', acc * 100, '%')

    return test_loss, acc, result_table


if __name__ == '__main__':
    main('model_q8.pth')
