import pandas as pd

df = pd.read_csv("test1.csv")
df.corr().to_csv("testout_1.4.csv", index=False)
