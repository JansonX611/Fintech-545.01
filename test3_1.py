import numpy as np
import pandas as pd

A = pd.read_csv("testout_1.3.csv").values
A = (A + A.T) / 2
d0 = np.diag(A).copy()

w, V = np.linalg.eigh(A)
w = np.maximum(w, 0.0)
B = V @ np.diag(w) @ V.T
B = (B + B.T) / 2

dB = np.diag(B)
s = np.sqrt(np.where(dB > 0, d0 / dB, 1.0))
S = np.diag(s)
X = S @ B @ S
X = (X + X.T) / 2

pd.DataFrame(X).to_csv("testout_3.1.csv", index=False)
