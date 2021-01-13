import torch

from core.abstract_model_wrapper import AbstractModelWrapper
from core.amino_utils import ClassificationMode
from models.model import Model


class ModelWrapper(Model, AbstractModelWrapper):

    def __init__(self, target_pos: int, target_class: int, mode: ClassificationMode):
        super(Model, self).__init__()
        super(ModelWrapper, self).__init__()
        self.__target_pos = target_pos
        self.__mode = mode
        self.__target_class = target_class

    def forward(self, x):
        out = super(ModelWrapper, self).forward(x)
        if self.__mode == ClassificationMode.Q3:
            index = torch.tensor([2, 1, 1, 0, 0, 0, 2, 2]).unsqueeze(0).unsqueeze(0)
            out = torch.zeros(out.shape).scatter_add(2, index.expand_as(out), out)
        return sum(torch.narrow(torch.narrow(torch.narrow(out, 1, self.__target_pos, 1), 2, self.__target_class, 1),
                                0, 0, 1))
