import numpy as np


train_path = 'data/cullpdb+profile_6133_filtered.npy'
train_dest = 'data/train.npz'
v_dest = 'data/v.npz'
test_path = 'data/cb513+profile_split1.npy'
test_dest = 'data/test.npz'


def init():
    print('Preparing dataset from cullpdb package...')

    x, y, seq_len = parse_dataset(train_path)
    x_train, y_train, seq_len_train = x[:4900], y[:4900], seq_len[:4900]
    x_v, y_v, seq_len_v = x[4900:5000], y[4900:5000], seq_len[4900:5000]
    x_test, y_test, seq_len_test = x[5000:], y[5000:], seq_len[5000:]
    np.savez_compressed(train_dest, X=x_train, y=y_train, seq_len=seq_len_train)
    np.savez_compressed(test_dest, X=x_test, y=y_test, seq_len=seq_len_test)
    np.savez_compressed(v_dest, X=x_v, y=y_v, seq_len=seq_len_v)

    print('done!')


def parse_dataset(path):
    data = np.load(path)
    data = data.reshape((-1, 700, 57))

    i = np.append(np.arange(21), np.append(np.arange(31, 33), np.arange(35, 56)))
    X = data[:, :, i]
    X = X.transpose(0, 2, 1)
    X = X.astype('float32')

    y = data[:, :, 22:30]
    y = np.array([np.dot(yi, np.arange(8)) for yi in y])
    y = y.astype('float32')

    mask = data[:, :, 30] * -1 + 1
    seq_len = mask.sum(axis=1)
    seq_len = seq_len.astype('float32')

    return X, y, seq_len


if __name__ == '__main__':
    init()
