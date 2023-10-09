import numpy as np


# Define new functions
def protectedDiv(left, right):
    try:
        if abs(right) > 0 and np.isfinite(right):
            res = left / right
            if np.isreal(res):
                return res
            else:
                return np.inf
        else:
            return np.inf
    except ZeroDivisionError:
        return np.inf


def protectedLog(x):
    try:
        log = np.log(x)
        if np.isfinite(log):
            return log
        else:
            return np.inf
    except:
        return np.inf


def protectedExp(x):
    try:
        if x <= 20:
            res = np.exp(x)
        else:
            res = np.inf
        return res
    except:
        return np.inf


def protectedSqrt(x):
    try:
        if x >= 0 and np.isreal(x):
            res = np.sqrt(x)
            if np.isfinite(res):
                return res
            else:
                return np.inf
        else:
            return np.inf
    except:
        return np.inf


def powerReal(x, R):
    try:
        res = x ** R
        if np.isreal(res):
            return res
        else:
            return np.inf
    except:
        return np.inf
