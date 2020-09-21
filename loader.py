import numpy as np
from torch.utils.data import Dataset, DataLoader
from prepare_data import test_dest, train_dest
from feature import feature_apply


class Data(Dataset):

    def __init__(self, x, y, seq_len):
        self.X = x
        self.y = y.astype(int)
        self.seq_len = seq_len.astype(int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        seq_len = self.seq_len[idx]
        return x, y, seq_len


class DataProvider:

    def __init__(self, batch_size_train, batch_size_test):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    def __load(self):

        train_data = np.load(train_dest)
        x_train, y_train, seq_len_train = train_data['X'], train_data['y'], train_data['seq_len']

        test_data = np.load(test_dest)
        x_test, y_test, seq_len_test = test_data['X'], test_data['y'], test_data['seq_len']

        # x_train = feature_apply(x_train, seq_len_train).astype('float32')
        # x_test = feature_apply(x_test, seq_len_test).astype('float32')
        #
        # print(x_train.dtype)

        # y_train = DataProvider.get_3_y(y_train)
        # y_test = DataProvider.get_3_y(y_test)

        return x_train, y_train, seq_len_train, x_test, y_test, seq_len_test

    @staticmethod
    @np.vectorize
    def get_3_y(inp):
        if inp == 0 or inp == 6 or inp == 7:
            return 0
        elif inp == 2 or inp == 1:
            return 1
        else:
            return 2

    def get_data(self):
        x_train, y_train, seq_len_train, x_test, y_test, seq_len_test = self.__load()

        data_train = Data(x_train, y_train, seq_len_train)
        train_loader = DataLoader(data_train, batch_size=self.batch_size_train, shuffle=True)

        data_test = Data(x_test, y_test, seq_len_test)
        test_loader = DataLoader(data_test, batch_size=self.batch_size_test, shuffle=False)

        return train_loader, test_loader
