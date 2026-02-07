import pandas as pd
import numpy as np

df = pd.read_csv("test6.csv")
dates = df["Date"].iloc[1:].reset_index(drop=True)
px = df.drop(columns=["Date"]).values
ret = np.log(px[1:] / px[:-1])

out = pd.DataFrame(ret, columns=df.columns[1:])
out.insert(0, "Date", dates)
out.to_csv("testout_6.2.csv", index=False)
