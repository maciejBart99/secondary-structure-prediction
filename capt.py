import torch
import matplotlib
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from feature import get_amino
from test import get_structure_label

from loader import DataProvider
from model_wrapper import ModelWrapper


matplotlib.use('Agg')

data = DataProvider(1024, 512)
data, target, seq_len = list(data.get_data()[1])[0]

print('loaded explainable...')


def explain_model(seq, pos, class_num, target_pos, model_pth):
    model = ModelWrapper(target_pos, class_num)
    model.load_state_dict(torch.load(model_pth))
    model.eval()
    ig = IntegratedGradients(model)
    input_tensor = data[seq:seq + 1, :, :]
    input_tensor.requires_grad_()
    attributions, delta = ig.attribute(input_tensor, torch.zeros(input_tensor.shape), target=0,
                                       return_convergence_delta=True)
    plt.figure(figsize=(16, 4))
    plt.bar([get_amino(x) for x in range(21)] + ['C-e', 'N-e'] + [get_amino(x) + '-M' for x in range(21)],
            attributions.data.numpy()[0, :, pos].tolist())
    name = f'fig/{seq}-{target_pos}-{get_structure_label(class_num)}-{pos}.png'
    plt.savefig(name)

    return name
