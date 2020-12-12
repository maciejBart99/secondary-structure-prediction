import numpy as np

from core.abstract_class_adapter import AbstractClassAdapter


@np.vectorize
def map_q8_to_q3(q8):
    if q8 == 3 or q8 == 4 or q8 == 5:
        return 0
    elif q8 == 1 or q8 == 2:
        return 1
    else:
        return 2


class BasicQ3Adapter(AbstractClassAdapter):

    def transform(self, inp: np.ndarray, seq_len: np.ndarray) -> np.ndarray:
        return map_q8_to_q3(inp)