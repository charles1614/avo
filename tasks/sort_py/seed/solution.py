"""Seed solution: correct but deliberately slow."""


def sort_list(arr):
    out = list(arr)
    n = len(out)
    for i in range(n):
        for j in range(0, n - i - 1):
            if out[j] > out[j + 1]:
                out[j], out[j + 1] = out[j + 1], out[j]
    return out
