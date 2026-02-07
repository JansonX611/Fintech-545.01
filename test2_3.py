import pandas as pd
import numpy as np

cov = pd.read_csv("testout_2.2.csv").values
std = np.sqrt(np.diag(cov))
corr = cov / np.outer(std, std)
np.fill_diagonal(corr, 1.0)

pd.DataFrame(corr).to_csv("testout_2.3.csv", index=False)
