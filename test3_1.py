import numpy as np
import pandas as pd

A = pd.read_csv("testout_1.3.csv").values
A = (A + A.T) / 2

d = np.sqrt(np.diag(A))
inv = np.diag(1.0 / np.where(d > 0, d, 1.0))
C = inv @ A @ inv
C = (C + C.T) / 2

w, V = np.linalg.eigh(C)
w = np.maximum(w, 0.0)
B = V @ np.diag(w) @ V.T
B = (B + B.T) / 2

s = 1.0 / np.sqrt(np.where(np.diag(B) > 0, np.diag(B), 1.0))
Cp = np.diag(s) @ B @ np.diag(s)
Cp = (Cp + Cp.T) / 2
np.fill_diagonal(Cp, 1.0)

D = np.diag(d)
X = D @ Cp @ D
X = (X + X.T) / 2
np.fill_diagonal(X, np.diag(A))

pd.DataFrame(X).to_csv("testout_3.1.csv", index=False)
