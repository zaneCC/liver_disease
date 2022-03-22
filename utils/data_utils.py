import numpy as np
import constants
import config

def unify_lable(y):
    _y = []
    for i in y:
        if i == 1:
            _y.append(0)
        else:
            _y.append(1)

    return np.array(_y)

