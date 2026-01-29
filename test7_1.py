import pandas as pd
import numpy as np

df = pd.read_csv("test7_1.csv")
x = df.iloc[:, 0].to_numpy()

mu = x.mean()
sigma = x.std(ddof=1)

pd.DataFrame(
    {"mu": [mu], "sigma": [sigma]}
).to_csv("testout7_1.csv", index=False)
