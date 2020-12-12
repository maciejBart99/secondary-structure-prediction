import numpy as np
from torch.utils.data import Dataset, DataLoader
from core.abstract_features import AbstractFeatures
from core.abstract_class_adapter import AbstractClassAdapter


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

    def __init__(self, batch_size_train, batch_size_test,
                 train_dest: str, test_dest: str,
                 feature_adapter: AbstractFeatures = None,
                 class_adapter: AbstractClassAdapter = None):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.train_dest = train_dest
        self.test_dest = test_dest
        self.feature_adapter = feature_adapter
        self.class_adapter = class_adapter

    def __load(self):

        train_data = np.load(self.train_dest)
        x_train, y_train, seq_len_train = train_data['X'], train_data['y'], train_data['seq_len']

        test_data = np.load(self.test_dest)
        x_test, y_test, seq_len_test = test_data['X'], test_data['y'], test_data['seq_len']

        if self.feature_adapter is not None:
            x_test = self.feature_adapter.inverse_features(
                self.feature_adapter.apply_features(x_test, seq_len_test), seq_len_test).astype('float32')
            x_train = self.feature_adapter.inverse_features(
                self.feature_adapter.apply_features(x_train, seq_len_train), seq_len_train).astype('float32')

        if self.class_adapter is not None:
            y_train = self.class_adapter.transform(y_train, seq_len_train)
            y_test = self.class_adapter.transform(y_test, seq_len_test)

        return x_train, y_train, seq_len_train, x_test, y_test, seq_len_test

    def get_data(self):
        x_train, y_train, seq_len_train, x_test, y_test, seq_len_test = self.__load()

        data_train = Data(x_train, y_train, seq_len_train)
        train_loader = DataLoader(data_train, batch_size=self.batch_size_train, shuffle=True)

        data_test = Data(x_test, y_test, seq_len_test)
        test_loader = DataLoader(data_test, batch_size=self.batch_size_test, shuffle=False)

        return train_loader, test_loader
