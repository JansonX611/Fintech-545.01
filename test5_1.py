import numpy as np
import pandas as pd

n = 100000

cin_df = pd.read_csv("test5_1.csv")
cin = cin_df.to_numpy(dtype=float)

L = np.linalg.cholesky(cin)
z = np.random.normal(size=(n, cin.shape[0]))
x = z @ L.T

cout = np.cov(x, rowvar=False, ddof=1)

pd.DataFrame(cout, columns=cin_df.columns).to_csv("testout_5.1.csv", index=False)
