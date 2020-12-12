from enum import Enum

import numpy as np


class ClassificationMode(Enum):
    Q3 = 'q3'
    Q8 = 'q8'


class AminoUtils:

    @staticmethod
    def normalize(inp: np.ndarray):
        partial = inp - inp.min(axis=0)
        return partial / partial.max(axis=0)

    @staticmethod
    def get_amino(num: int):
        dc = {
            0: 'A', 1: 'C', 2: 'E', 3: 'D', 4: 'G', 5: 'F', 6: 'I', 7: 'H', 8: 'K',
            9: 'M', 10: 'L', 11: 'N', 12: 'Q',
            13: 'P', 14: 'S', 15: 'R', 16: 'T', 17: 'W', 18: 'V', 19: 'Y', 20: 'X'
        }

        if num in dc:
            return dc[num]
        else:
            return ' '

    @staticmethod
    def get_structure_label(num: int):
        dc = {
            0: 'L', 1: 'B', 2: 'E', 3: 'G', 4: 'I', 5: 'H', 6: 'S', 7: 'T',
        }

        if num in dc:
            return dc[num]
        else:
            return ' '

    @staticmethod
    def get_structure_label_q3(num: int):
        dc = {
            0: 'H', 1: 'E', 2: 'L'
        }

        if num in dc:
            return dc[num]
        else:
            return ' '

    @staticmethod
    def accuracy(out, target, seq_len):
        return np.array([np.equal(o[:l], t[:l]).sum() / l
                         for o, t, l in zip(out, target, seq_len)]).mean()
