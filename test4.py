import numpy as np
import pandas as pd

A = pd.read_csv("testout_3.1.csv").values
A = (A + A.T) / 2
n = A.shape[0]

L = np.zeros_like(A)

for i in range(n):
    s = np.dot(L[i, :i], L[i, :i])
    d = A[i, i] - s
    if d < 0:
        d = 0.0
    L[i, i] = np.sqrt(d)

    if L[i, i] > 0:
        for j in range(i + 1, n):
            s2 = np.dot(L[j, :i], L[i, :i])
            L[j, i] = (A[j, i] - s2) / L[i, i]
    else:
        L[i + 1 :, i] = 0.0

pd.DataFrame(L).to_csv("testout_4.1.csv", index=False)
