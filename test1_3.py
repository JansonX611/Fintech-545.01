import pandas as pd

df = pd.read_csv("test1.csv")
df.cov().to_csv("testout_1.3.csv", index=False)
