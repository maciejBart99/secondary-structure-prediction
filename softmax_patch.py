import torch


class ShapeMismatchException(Exception):
    pass


def softmax(inp: torch.Tensor, weights: torch.Tensor, dim=2):
    if inp.shape[dim] != weights.shape[0]:
        raise ShapeMismatchException()

    for i in range(inp.shape[dim]):
        inp[:, :, i] = inp[:, :, i] * weights[i]

    return torch.nn.functional.softmax(inp, dim=dim)
