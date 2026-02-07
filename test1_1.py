import pandas as pd
import numpy as np

df = pd.read_csv("test1.csv").dropna()
cov = np.cov(df.values, rowvar=False, bias=False)
pd.DataFrame(cov, columns=df.columns).to_csv("testout_1.1.csv", index=False)
