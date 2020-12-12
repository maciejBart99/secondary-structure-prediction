from dataclasses import dataclass
from types import new_class

import torch
import numpy as np
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader

from core.amino_utils import ClassificationMode
from core.test_manager import ModelDescriptor


@dataclass
class ExplainConfig:
    model: ModelDescriptor
    wrapper_class: new_class
    mode: ClassificationMode = ClassificationMode.Q8
    window: int = 8


@dataclass
class ExplainResult:
    seq_from: int
    seq_to: int
    data: np.ndarray


class ExplainManager:

    def __init__(self, config: ExplainConfig, loader: DataLoader):
        self.config = config
        self.loader = loader
        data, _, seq_len = list(self.loader)[0]
        self.data = data
        self.seq_len = seq_len

    def explain(self, seq: int, class_num: int, target_pos: int):

        model = self.config.wrapper_class(target_pos, class_num, self.config.mode)
        model.load_state_dict(torch.load(self.config.model.path))
        torch.autograd.set_detect_anomaly(True)
        model.eval()
        ig = IntegratedGradients(model)
        input_tensor = self.data[seq:seq + 1, :, :]
        input_tensor.requires_grad_()
        attributions, delta = ig.attribute(input_tensor, torch.zeros(input_tensor.shape), target=0,
                                           return_convergence_delta=True)
        right_limit = min(self.seq_len[seq], target_pos + self.config.window + 1)
        left_limit = max(0, target_pos - self.config.window)

        return ExplainResult(left_limit, right_limit, attributions.data.numpy()[0, :, left_limit:right_limit])


