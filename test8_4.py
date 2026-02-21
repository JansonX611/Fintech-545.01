import pandas as pd
import numpy as np
import scipy.stats as st

ALPHA = 0.05

def es_normal(mu: float, sigma: float, alpha: float = ALPHA) -> float:
    z = st.norm.ppf(alpha)
    phi = st.norm.pdf(z)
    return -mu + sigma * phi / alpha

def main():
    x = pd.read_csv("test7_1.csv").iloc[:, 0].to_numpy(dtype=float)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))

    out = pd.DataFrame({
        "ES Absolute": [es_normal(mu, sigma)],
        "ES Diff from Mean": [es_normal(0.0, sigma)],
    })
    out.to_csv("testout8_4.csv", index=False)

if __name__ == "__main__":
    main()