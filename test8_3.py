import numpy as np
import pandas as pd
from scipy.stats import t

cin = pd.read_csv("test7_2.csv")
x = cin.iloc[:, 0].to_numpy(dtype=float)

df, loc, scale = t.fit(x)

u = np.random.rand(10000)
sim = t.ppf(u, df, loc=loc, scale=scale)

var_abs = -np.quantile(sim, 0.05)
var_dm = -np.quantile(sim - float(np.mean(sim)), 0.05)

out = pd.DataFrame({"VaR Absolute": [var_abs], "VaR Diff from Mean": [var_dm]})
out.to_csv("testout_8.3.csv", index=False)
