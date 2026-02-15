import numpy as np
import pandas as pd

n = 100000

cin_df = pd.read_csv("test5_3.csv")
a = cin_df.to_numpy(dtype=float)

def sym(m):
    return (m + m.T) / 2.0

def proj_psd(m):
    w, v = np.linalg.eigh(sym(m))
    w = np.maximum(w, 0.0)
    return v @ np.diag(w) @ v.T

def higham_nearest_psd(m, iters=100, tol=1e-10):
    y = sym(m)
    delta = np.zeros_like(y)
    for _ in range(iters):
        r = y - delta
        x = proj_psd(r)
        delta = x - r
        y_new = sym(x)
        if np.linalg.norm(y_new - y, ord="fro") <= tol:
            y = y_new
            break
        y = y_new
    return sym(y)

cin_fix = higham_nearest_psd(a)

w, v = np.linalg.eigh(cin_fix)
w = np.maximum(w, 0.0)
A = v @ np.diag(np.sqrt(w))

z = np.random.normal(size=(n, cin_fix.shape[0]))
x = z @ A.T

cout = np.cov(x, rowvar=False, ddof=1)

pd.DataFrame(cout, columns=cin_df.columns).to_csv("testout_5.4.csv", index=False)
