from typing import List

import torch
import numpy as np

from core.amino_utils import ClassificationMode, AminoUtils
from core.abstract_output import AbstractOutput
from core.abstract_class_adapter import AbstractClassAdapter
from core.loader import DataLoader
from model import CrossEntropy


class ModelDescriptor:

    def __init__(self, path: str, _class: torch.nn.Module, mode: ClassificationMode):
        self.path = path
        self.cls = _class
        self.mode = mode


class TestConfig:

    def __init__(self, model: ModelDescriptor, output: AbstractOutput = None, log=False,
                 mode=ClassificationMode.Q8, test_batch=512, dump_predictions=None):
        self.model = model
        self.output = output
        self.log = log
        self.mode = mode
        self.test_batch = test_batch
        self.dump_predictions = dump_predictions


class TestResult:
    class TestRecord:

        def __init__(self, _id: int, acc: float, _len: int, seq_str: str, pred: str, target: str,
                     possibility: List[int]):
            self.id = _id
            self.acc = acc
            self.len = _len
            self.seq_str = seq_str
            self.pred = pred
            self.target = target
            self.possibility = possibility

    def __init__(self, acc: float, loss: float, records: List[TestRecord]):
        self.acc = acc
        self.loss = loss
        self.records = records


class TestManager:

    def __init__(self, config: TestConfig, test_loader: DataLoader, class_adapter: AbstractClassAdapter = None):
        self.__config = config
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.test_loader = test_loader
        self.class_adapter = class_adapter

    def test(self):
        model = self.__config.model.cls.to(self.device)
        model.load_state_dict(torch.load(self.__config.model.path))
        loss_function = CrossEntropy()

        model.eval()
        total_accuracy = 0
        total_loss = 0
        result: List[TestResult.TestRecord] = []
        length = len(self.test_loader)

        with torch.no_grad():
            total_predictions = []
            for i, (data, target, seq_len) in enumerate(self.test_loader):
                data, target, seq_len = data.to(self.device), target.to(self.device), seq_len.to(self.device)
                out = model(data)

                total_loss += loss_function(out, target, seq_len).cpu().data.numpy()
                cumulative_accuracy = 0

                for ind in range(min(self.__config.test_batch, seq_len.shape[0])):
                    class_label_decoder = AminoUtils.get_structure_label if self.__config.mode == ClassificationMode.Q3 \
                        else AminoUtils.get_structure_label_q3
                    expected = target[ind, :seq_len[ind]].data.numpy()
                    expected = expected if self.__config.mode == self.__config.model.mode \
                        else self.class_adapter.transform(expected, seq_len[ind])
                    predictions = out[ind, :seq_len[ind], :].data.numpy().argmax(axis=1)
                    predictions = predictions if self.__config.mode == self.__config.model.mode \
                        else self.class_adapter.transform(predictions, seq_len[ind])

                    total_predictions.append(
                        (out[ind, :, :].data.numpy() if self.__config.mode == self.__config.model.mode else
                         self.class_adapter.transform(out[ind, :, :].data.numpy(),
                         seq_len[ind])).argmax(axis=1).reshape((1, 700)))

                    ac = AminoUtils.accuracy(predictions.reshape((1, -1)), expected.reshape((1, -1)),
                                             seq_len[ind:ind + 1])
                    cumulative_accuracy += ac
                    seq_str = ''.join([AminoUtils.get_amino(x) for x in
                                       data[ind, :21, :seq_len[ind]].data.numpy().argmax(axis=0).tolist()])
                    pred_str = ''.join([class_label_decoder(x) for x in predictions.tolist()])
                    target_str = ''.join([class_label_decoder(x) for x in expected.tolist()])

                    if self.__config.log and self.__config.output is not None:
                        self.__config.output.write(f'Ind {ind} Accuracy {ac:.3f} len {seq_len[ind]}')
                        self.__config.output.write(seq_str)
                        self.__config.output.write(pred_str)
                        self.__config.output.write(target_str)
                    possibility = out[ind, :seq_len[ind], :].data.numpy()
                    if self.__config.mode == ClassificationMode.Q3:
                        pos = np.zeros((possibility.shape[0], 3))
                        pos[:, 0] = possibility[:, 3:6].sum(1)
                        pos[:, 1] = possibility[:, 1:3].sum(1)
                        pos[:, 2] = possibility[:, 6:].sum(1) + possibility[:, 0]
                        possibility = pos
                    result.append(TestResult.TestRecord(ind, ac, seq_len[ind], seq_str, pred_str, target_str,
                                                        possibility.tolist()))
                cumulative_accuracy /= target.shape[0]
                total_accuracy += cumulative_accuracy
            if self.__config.dump_predictions:
                np.save(self.__config.dump_predictions, np.concatenate(total_predictions))
        total_accuracy /= length

        return TestResult(total_accuracy, total_loss, result)
