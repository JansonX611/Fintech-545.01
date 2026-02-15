import numpy as np
import pandas as pd

n = 100000
pct = 0.99

cin_df = pd.read_csv("test5_2.csv")
cin = cin_df.to_numpy(dtype=float)

cin = (cin + cin.T) / 2.0
w, v = np.linalg.eigh(cin)
idx = np.argsort(w)[::-1]
w = w[idx]
v = v[:, idx]

total = w.sum()
cum = np.cumsum(w) / total
k = int(np.searchsorted(cum, pct) + 1)

wk = np.maximum(w[:k], 0.0)
vk = v[:, :k]
B = vk @ np.diag(np.sqrt(wk))

z = np.random.normal(size=(n, k))
x = z @ B.T

cout = np.cov(x, rowvar=False, ddof=1)

pd.DataFrame(cout, columns=cin_df.columns).to_csv("testout_5.5.csv", index=False)
