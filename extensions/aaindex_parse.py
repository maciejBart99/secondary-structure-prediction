from typing import Dict

import numpy as np

from core.amino_utils import AminoUtils


def aaindex_parse(path: str) -> Dict[str, Dict[str, float]]:
    result = dict()
    with open(path, 'r') as f:
        state = ""
        feature_name = ""
        feature = dict()
        lines = f.readlines()
        for line in lines:
            if line.startswith("H "):
                state = 'FEAT_START'
            elif line.startswith("D "):
                feature_name = line.replace("D ", "")
                state = 'FEAT_NAME_READ'
            elif line.startswith("I "):
                state = 'FEAT_READ'
            elif line.startswith("//"):
                feature_name = ''
                feature = dict()
            elif state == 'FEAT_READ':
                values = line.strip().split(' ')
                values = list(filter(lambda x: x != '', values))
                values = list(map(lambda x: x if x != 'NA' else '0', values))
                feature['A'] = float(values[0])
                feature['R'] = float(values[1])
                feature['N'] = float(values[2])
                feature['D'] = float(values[3])
                feature['C'] = float(values[4])
                feature['Q'] = float(values[5])
                feature['E'] = float(values[6])
                feature['G'] = float(values[7])
                feature['H'] = float(values[8])
                feature['I'] = float(values[9])
                state = 'FEAT_1_READ'
            elif state == 'FEAT_1_READ':
                values = line.strip().split(' ')
                values = list(filter(lambda x: x != '', values))
                values = list(map(lambda x: x if x != 'NA' else '0', values))
                feature['L'] = float(values[0])
                feature['K'] = float(values[1])
                feature['M'] = float(values[2])
                feature['F'] = float(values[3])
                feature['P'] = float(values[4])
                feature['S'] = float(values[5])
                feature['T'] = float(values[6])
                feature['W'] = float(values[7])
                feature['Y'] = float(values[8])
                feature['V'] = float(values[9])
                feature['X'] = 0
                state = 'READ_END'
                result[feature_name] = feature
    return result


def aaindex_parse_with_sorting_and_normalization(path: str) -> Dict[str, np.ndarray]:
    parsed = aaindex_parse(path)
    sorted_features = {key: np.array([float(x[1]) for x in sorted(list(value.items()), key=lambda x: x[0])])[:21] for (key, value) in parsed.items()}
    r_dict = dict()
    for key in sorted_features:
        r_dict[key] = AminoUtils.normalize(sorted_features[key])

    return r_dict