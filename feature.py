import numpy as np


def get_vaules(d):
    return np.array(list(d.values())[:21])


electric_charge = get_vaules({
  'A': 0.0,
  'C': 0.0,
  'D': -1.0,
  'E': -1.0,
  'F': 0.0,
  'G': 0.0,
  'H': 1.0,
  'I': 0.0,
  'K': 1.0,
  'L': 0.0,
  'M': 0.0,
  'N': 0.0,
  'P': 0.0,
  'Q': 0.0,
  'R': 1.0,
  'S': 0.0,
  'T': 0.0,
  'V': 0.0,
  'W': 0.0,
  'Y': 0.0,
  'X': 0.0,
  '.': 0.0,
  '_': 0.0
})

hydrophobicity_score = get_vaules({
  'A': 0.159,
  'C': 0.778,
  'D': -1.289,
  'E': -1.076,
  'F': 1.437,
  'G': -0.131,
  'H': -0.553,
  'I': 1.388,
  'K': -1.504,
  'L': 1.236,
  'M': 1.048,
  'N': -0.866,
  'P': -0.104,
  'Q': -0.836,
  'R': -1.432,
  'S': -0.549,
  'T': -0.292,
  'V': 1.064,
  'W': 1.046,
  'Y': 0.476 ,
  'X':-0.078,
  '.': -0.078,
  '_': -0.078
})

polarity = get_vaules({
  'A': 0.0,
  'C': 0.0,
  'D': 0.0,
  'E': 0.0,
  'F': 0.0,
  'G': 0.0,
  'H': 0.0,
  'I': 0.0,
  'K': 0.0,
  'L': 0.0,
  'M': 0.0,
  'N': 1.0,
  'P': 0.0,
  'Q': 1.0,
  'R': 0.0,
  'S': 1.0,
  'T': 1.0,
  'V': 0.0,
  'W': 0.0,
  'Y': 0.0,
  'X': 0.0,
  '.': 0.0,
  '_': 0.0
})

volume = get_vaules({
  'A': 92.0,
  'C': 106.0,
  'D': 125.0,
  'E': 155.0,
  'F': 203.0,
  'G': 66.0,
  'H': 167.0,
  'I': 169.0,
  'K': 171.0,
  'L': 168.0,
  'M': 171.0,
  'N': 135.0,
  'P': 129.0,
  'Q': 161.0,
  'R': 225.0,
  'S': 99.0,
  'T': 122.0,
  'V': 142.0,
  'W': 240.0,
  'Y': 203.0,
  'X': 1043.768,
  '.': 1043.768,
  '_': 1043.768
})

solvent_accessibility = get_vaules({
  'S': [0.70, 0.20, 0.10],
  'T': [0.71, 0.16, 0.13],
  'A': [0.48, 0.35, 0.17],
  'G': [0.51, 0.36, 0.13],
  'P': [0.78, 0.13, 0.09],
  'C': [0.32, 0.54, 0.14],
  'D': [0.81, 0.09, 0.10],
  'E': [0.93, 0.04, 0.03],
  'Q': [0.81, 0.10, 0.09],
  'N': [0.82, 0.10, 0.08],
  'L': [0.41, 0.49, 0.10],
  'I': [0.39, 0.47, 0.14],
  'V': [0.40, 0.50, 0.10],
  'M': [0.44, 0.20, 0.36],
  'F': [0.42, 0.42, 0.16],
  'Y': [0.67, 0.20, 0.13],
  'W': [0.49, 0.44, 0.07],
  'K': [0.93, 0.02, 0.05],
  'R': [0.84, 0.05, 0.11],
  'H': [0.66, 0.19, 0.15],
  'X': [0.63, 0.25, 0.12],
  '.': [0.63, 0.25, 0.12],
  '_': [0.63, 0.25, 0.12]
})


def get_amino(num):
    dc = {
        0: 'A', 1: 'C', 2: 'E', 3: 'D', 4: 'G', 5: 'F', 6: 'I', 7: 'H', 8: 'K', 9: 'M', 10: 'L', 11: 'N', 12: 'Q',
        13: 'P', 14: 'S', 15: 'R', 16: 'T', 17: 'W', 18: 'V', 19: 'Y', 20: 'X'
    }

    if num in dc:
        return dc[num]
    else:
        return ' '


def feature_layer(inp: np.ndarray, seq_len: np.ndarray):
    result = np.zeros((inp.shape[0], 7 + 42 + 2, inp.shape[2]))
    result[:, 9:30, :] = inp[:, 23:, :]
    result[:, 30:, :] = inp[:, :21, :]
    result[:, 7:9, :] = inp[:, 21:23, :]
    # result[:, 0, :] = np.vectorize(lambda x: electric_charge[x])(inp[:, :21, :].argmax(1))
    # result[:, 1, :] = np.vectorize(lambda x: hydrophobicity_score[x])(inp[:, :21, :].argmax(1))
    # result[:, 2, :] = np.vectorize(lambda x: polarity[x])(inp[:, :21, :].argmax(1))
    # result[:, 3, :] = np.vectorize(lambda x: volume[x])(inp[:, :21, :].argmax(1))
    # result[:, 4, :] = np.vectorize(lambda x: solvent_accessibility[x, 0])(inp[:, :21, :].argmax(1))
    # result[:, 5, :] = np.vectorize(lambda x: solvent_accessibility[x, 1])(inp[:, :21, :].argmax(1))
    # result[:, 6, :] = np.vectorize(lambda x: solvent_accessibility[x, 2])(inp[:, :21, :].argmax(1))

    for i in range(inp.shape[0]):
        print(i)
        for j in range(int(seq_len[i])):
            result[i, 0, j] = np.dot(inp[i, 23:, j], electric_charge)
            result[i, 1, j] = np.dot(inp[i, 23:, j], hydrophobicity_score)
            result[i, 2, j] = np.dot(inp[i, 23:, j], polarity)
            result[i, 3, j] = np.dot(inp[i, 23:, j], volume)
            result[i, 4, j] = np.dot(inp[i, 23:, j], solvent_accessibility[:, 0])
            result[i, 5, j] = np.dot(inp[i, 23:, j], solvent_accessibility[:, 1])
            result[i, 6, j] = np.dot(inp[i, 23:, j], solvent_accessibility[:, 2])

    return result
