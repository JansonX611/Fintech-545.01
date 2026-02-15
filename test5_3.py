import numpy as np
import pandas as pd

n = 100000

cin_df = pd.read_csv("test5_3.csv")
cin = cin_df.to_numpy(dtype=float)

cin = (cin + cin.T) / 2.0
w, v = np.linalg.eigh(cin)
w = np.maximum(w, 0.0)
cin_fix = v @ np.diag(w) @ v.T
cin_fix = (cin_fix + cin_fix.T) / 2.0

w2, v2 = np.linalg.eigh(cin_fix)
w2 = np.maximum(w2, 0.0)
A = v2 @ np.diag(np.sqrt(w2))

z = np.random.normal(size=(n, cin.shape[0]))
x = z @ A.T

cout = np.cov(x, rowvar=False, ddof=1)

pd.DataFrame(cout, columns=cin_df.columns).to_csv("testout_5.3.csv", index=False)
