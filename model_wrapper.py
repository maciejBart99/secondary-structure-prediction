import torch

from model import Model


def map_q8_to_q3(q8):
    if q8 == 3 or q8 == 4 or q8 == 5:
        return 0
    elif q8 == 1 or q8 == 2:
        return 1
    else:
        return 2


class ModelWrapper(Model):

    def __init__(self, target_pos, target_class, mode):
        self.__target_pos = target_pos
        self.__target_class = target_class
        self.__mode = mode
        super().__init__()

    def forward(self, x):
        out = super().forward(x)
        if self.__mode == 'q3':
            index = torch.tensor([2, 1, 1, 0, 0, 0, 2, 2]).unsqueeze(0).unsqueeze(0)
            out = torch.zeros(out.shape).scatter_add(2, index.expand_as(out), out)
        return sum(torch.narrow(torch.narrow(torch.narrow(out, 1, self.__target_pos, 1), 2, self.__target_class, 1),
                                0, 0, 1))
