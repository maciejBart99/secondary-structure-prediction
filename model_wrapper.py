import torch

from model import Model


class ModelWrapper(Model):

    def __init__(self, target_pos, target_class):
        self.__target_pos = target_pos
        self.__target_class = target_class
        super().__init__()

    def forward(self, x):
        out = super().forward(x)
        return sum(torch.narrow(torch.narrow(torch.narrow(out, 1, self.__target_pos, 1), 2, self.__target_class, 1),
                                0, 0, 1))
