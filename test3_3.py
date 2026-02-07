import numpy as np
import pandas as pd

A = pd.read_csv("testout_1.3.csv").values
A = (A + A.T) / 2

d = np.sqrt(np.diag(A))
inv = np.diag(1.0 / np.where(d > 0, d, 1.0))
C = inv @ A @ inv
C = (C + C.T) / 2

X = C.copy()
dS = np.zeros_like(C)
tol = 1e-9
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

D = np.diag(d)
Cov = D @ X @ D
Cov = (Cov + Cov.T) / 2
np.fill_diagonal(Cov, np.diag(A))

pd.DataFrame(Cov).to_csv("testout_3.3.csv", index=False)
