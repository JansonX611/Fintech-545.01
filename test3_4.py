import numpy as np
import pandas as pd

A = pd.read_csv("testout_1.4.csv").values
A = (A + A.T) / 2

X = A.copy()
dS = np.zeros_like(A)
tol = 1e-10
max_iter = 500

for _ in range(max_iter):
    Y = X - dS
    w, V = np.linalg.eigh((Y + Y.T) / 2)
    w = np.maximum(w, 0.0)
    Xp = V @ np.diag(w) @ V.T
    Xp = (Xp + Xp.T) / 2
    dS = Xp - Y

    Xn = Xp.copy()
    np.fill_diagonal(Xn, 1.0)
    Xn = (Xn + Xn.T) / 2

    if np.linalg.norm(Xn - X, ord="fro") / max(1.0, np.linalg.norm(X, ord="fro")) < tol:
        X = Xn
        break
    X = Xn

pd.DataFrame(X).to_csv("testout_3.4.csv", index=False)
