import numpy as np
import pandas as pd
from scipy.stats import t

cin = pd.read_csv("test7_2.csv")
x = cin.iloc[:, 0].to_numpy(dtype=float)

df, loc, scale = t.fit(x)

var_abs = -t.ppf(0.05, df, loc=loc, scale=scale)
var_dm = -t.ppf(0.05, df, loc=0.0, scale=scale)

out = pd.DataFrame({"VaR Absolute": [var_abs], "VaR Diff from Mean": [var_dm]})
out.to_csv("testout_8.2.csv", index=False)
