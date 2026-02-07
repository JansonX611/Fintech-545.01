import numpy as np
import pandas as pd

A = pd.read_csv("testout_1.4.csv").values
A = (A + A.T) / 2

w, V = np.linalg.eigh(A)
w = np.maximum(w, 0.0)
B = V @ np.diag(w) @ V.T
B = (B + B.T) / 2

s = 1.0 / np.sqrt(np.where(np.diag(B) > 0, np.diag(B), 1.0))
X = np.diag(s) @ B @ np.diag(s)
X = (X + X.T) / 2
np.fill_diagonal(X, 1.0)

pd.DataFrame(X).to_csv("testout_3.2.csv", index=False)
