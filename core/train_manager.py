import os

from typing import List

import torch

from core.abstract_class_adapter import AbstractClassAdapter
from core.abstract_output import AbstractOutput
from core.test_manager import ModelDescriptor, TestManager, TestConfig
from core.loader import DataProvider
from model import CrossEntropy


class TrainResult:

    def __init__(self, acc: float, acc_history: List[float], loss: float):
        self.acc = acc
        self.acc_history = acc_history
        self.loss = loss


class TrainConfig:

    def __init__(self, epochs: int, output: AbstractOutput, model: ModelDescriptor, save_path: str):
        self.epochs = epochs
        self.output = output
        self.model = model
        self.save_path = save_path


class TrainManager:

    def __init__(self, config: TrainConfig, provider: DataProvider, class_adapter: AbstractClassAdapter=None):
        torch.manual_seed(1)
        self.__config = config
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.data = provider
        self.class_adapter = class_adapter

    def train(self):
        model = self.__config.model.cls.to(self.device)
        loss_function = CrossEntropy()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)

        train_loader, test_loader = self.data.get_data()

        for e in range(self.__config.epochs):
            train_loss = self.__train_epoch(model, train_loader, loss_function, optimizer)
            save_to = os.path.join(self.__config.save_path, f'model_{e}.pth')
            torch.save(model.state_dict(), save_to)
            self.__config.model.path = save_to
            test_config = TestConfig(
                self.__config.model,
                mode=self.__config.model.mode
            )
            test_manager = TestManager(test_config, test_loader, self.class_adapter)
            test_result = test_manager.test()
            if self.__config.output is not None:
                self.__config.output.write(f'[{e:3d}/{self.__config.epochs:3d}]' +
                                           f' train_loss:{train_loss:.2f},' +
                                           f' test_loss:{test_result.loss:.2f}, acc:{test_result.acc:.3f}')

    def __train_epoch(self, model, train_loader, loss_function, optimizer) -> float:
        model.train()
        train_loss = 0
        length = len(train_loader)
        for batch_idx, (data, target, seq_len) in enumerate(train_loader):
            data, target, seq_len = data.to(self.device), target.to(self.device), seq_len.to(self.device)
            optimizer.zero_grad()
            data = data.float()
            out = model(data)
            loss = loss_function(out, target, seq_len)
            loss.backward()
            optimizer.step()
            if self.__config.output is not None:
                self.__config.output.write(f'Batch {batch_idx} loss {loss.item():.2f}')
            train_loss += loss.item()

        train_loss /= length

        return train_loss
