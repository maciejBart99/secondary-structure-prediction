import numpy as np
from sympy import *


class MathUtils:

    @staticmethod
    def find_reverse_transformation_matrix(A: np.ndarray) -> np.ndarray:
        params = np.random.rand(21 - A.shape[0], A.shape[0])
        m_f_matrix = Matrix((np.concatenate((A, np.eye(A.shape[0])), axis=1)).tolist())
        m_f_matrix_rr, ind = m_f_matrix.rref()
        m_f_matrix_rr = np.array(m_f_matrix_rr.tolist()).astype(np.float)
        m_f_inv = np.ones((A.shape[1], A.shape[0]))
        m_f_inv[ind, :] = m_f_matrix_rr[:, -A.shape[0]:] - np.matmul(
            m_f_matrix_rr[:, [x for x in range(21) if x not in ind]], params)
        return m_f_inv
