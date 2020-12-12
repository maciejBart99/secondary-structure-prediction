from typing import List


def rescue_accuracy_from_log(file: str) -> List[float]:
    res = []
    with open(file, 'r') as f:
        for l in f.readlines():
            if l.startswith("[Acc]"):
                res.append(float(l.split(' ')[2]))
    return res
