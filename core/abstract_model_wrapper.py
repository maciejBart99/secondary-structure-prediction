from abc import ABC, abstractmethod

from core.amino_utils import ClassificationMode


class AbstractModelWrapper(ABC):

    def __init__(self, target_pos: int, target_class: int, mode: ClassificationMode):
        super().__init__()
        self.__target_pos = target_pos
        self.__target_class = target_class
        self.__mode = mode

    @abstractmethod
    def forward(self, x):
        pass
