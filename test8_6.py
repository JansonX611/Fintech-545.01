import pandas as pd
import numpy as np
import scipy.stats as st

ALPHA = 0.05
N_SIM = 10000

def sample_es(x: np.ndarray, alpha: float = ALPHA) -> float:
    q = np.quantile(x, alpha)
    return -float(np.mean(x[x <= q]))

def main():
    x = pd.read_csv("test7_2.csv").iloc[:, 0].to_numpy(dtype=float)
    df, loc, scale = st.t.fit(x)

    rng = np.random.default_rng(2)
    u = rng.random(N_SIM)
    sim = st.t.ppf(u, df, loc=loc, scale=scale)

    out = pd.DataFrame({
        "ES Absolute": [sample_es(sim)],
        "ES Diff from Mean": [sample_es(sim - float(np.mean(sim)))],
    })
    out.to_csv("testout8_6.csv", index=False)

if __name__ == "__main__":
    main()