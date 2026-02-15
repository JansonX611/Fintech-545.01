import numpy as np
import pandas as pd

n = 100000

cin_df = pd.read_csv("test5_2.csv")
cin = cin_df.to_numpy(dtype=float)

w, v = np.linalg.eigh((cin + cin.T) / 2.0)
w = np.maximum(w, 0.0)
A = v @ np.diag(np.sqrt(w))

z = np.random.normal(size=(n, cin.shape[0]))
x = z @ A.T

cout = np.cov(x, rowvar=False, ddof=1)

pd.DataFrame(cout, columns=cin_df.columns).to_csv("testout_5.2.csv", index=False)
