import numpy as np


def group_counts(arr):
    d = np.ones(arr.size, dtype=int)
    d[1:] = (arr[:-1] != arr[1:]).astype(int)

    return np.diff(np.where(np.append(d, 1))[0])


def group_offsets(arr):
    d = np.ones(arr.size, dtype=int)
    d[1:] = (arr[:-1] != arr[1:]).astype(int)
    idx = np.where(np.append(d, 1))[0]

    return zip(idx, idx[1:])
