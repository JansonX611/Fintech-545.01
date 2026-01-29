import pandas as pd
from scipy.stats import t

df = pd.read_csv("test7_2.csv")
x = df.iloc[:, 0].to_numpy()

nu, mu, sigma = t.fit(x)

pd.DataFrame(
    {"mu": [mu], "sigma": [sigma], "nu": [nu]}
).to_csv("testout7_2.csv", index=False)
