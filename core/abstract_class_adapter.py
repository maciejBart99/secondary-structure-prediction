from abc import ABC, abstractmethod
import numpy as np


class AbstractClassAdapter(ABC):

    @abstractmethod
    def transform(self, inp: np.ndarray, seq_len: np.ndarray) -> np.ndarray:
        pass
