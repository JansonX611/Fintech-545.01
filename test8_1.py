import numpy as np
import pandas as pd
from scipy.stats import norm

cin = pd.read_csv("test7_1.csv")
x = cin.iloc[:, 0].to_numpy(dtype=float)

mu = float(np.mean(x))
sigma = float(np.std(x, ddof=1))

var_abs = -norm.ppf(0.05, loc=mu, scale=sigma)
var_dm = -norm.ppf(0.05, loc=0.0, scale=sigma)

out = pd.DataFrame({"VaR Absolute": [var_abs], "VaR Diff from Mean": [var_dm]})
out.to_csv("testout_8.1.csv", index=False)
