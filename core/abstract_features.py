from abc import abstractmethod, ABC
import numpy as np


class AbstractFeatures(ABC):

    @abstractmethod
    def apply_features(self, inp: np.ndarray, seq_len: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_features(self, inp: np.ndarray, seq_len: np.ndarray) -> np.ndarray:
        pass
