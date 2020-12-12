import numpy as np

from core.abstract_features import AbstractFeatures
from core.math_utils import MathUtils


class BasicAdapter(AbstractFeatures):

    def __init__(self, features: np.ndarray, disable_inv=False):
        self.M = features
        self.disable_inv = disable_inv
        if not disable_inv:
            self.M_inv = MathUtils.find_reverse_transformation_matrix(features)

    def apply_features(self, inp: np.ndarray, seq_len: np.ndarray) -> np.ndarray:
        s = self.M.shape[0]
        result = np.zeros((inp.shape[0], 2 * s + 2, inp.shape[2]))
        result[:, 2 * s, :] = inp[:, 21, :]
        result[:, 2 * s + 1, :] = inp[:, 22, :]
        for i in range(inp.shape[0]):
            for j in range(int(seq_len[i])):
                result[i, :s, j] = np.matmul(self.M, inp[i, :21, j])
                result[i, s:2 * s, j] = np.matmul(self.M, inp[i, 23:, j])

        return result

    def inverse_features(self, inp: np.ndarray, seq_len: np.ndarray) -> np.ndarray:
        if self.disable_inv:
            return inp
        else:
            print('inv')
            s = self.M.shape[0]
            result = np.zeros((inp.shape[0], 44, inp.shape[2]))
            result[:, 21, :] = inp[:, 2 * s, :]
            result[:, 22, :] = inp[:, 2 * s + 1, :]
            for i in range(inp.shape[0]):
                for j in range(int(seq_len[i])):
                    result[i, :21, j] = np.matmul(self.M_inv, inp[i, :s, j])
                    result[i, 23:, j] = np.matmul(self.M_inv, inp[i, s:2 * s, j])

            return result
