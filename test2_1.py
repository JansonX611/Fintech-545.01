import pandas as pd
import numpy as np

df = pd.read_csv("test2.csv")
lam = 0.94
n = len(df)
w = np.array([(1 - lam) * lam**(n - i - 1) for i in range(n)])
w = w / w.sum()

mean = (df.values * w[:, None]).sum(axis=0)
pd.DataFrame(mean.reshape(1, -1), columns=df.columns).to_csv("testout_2.1.csv", index=False)
