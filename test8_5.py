import pandas as pd
import numpy as np
import scipy.stats as st

ALPHA = 0.05

def es_t(df: float, loc: float, scale: float, alpha: float = ALPHA) -> float:
    a = st.t.ppf(alpha, df)
    pdf = st.t.pdf(a, df)
    cond_mean = -((df + a * a) / (df - 1.0)) * (pdf / alpha)
    return -(loc + scale * cond_mean)

def main():
    x = pd.read_csv("test7_2.csv").iloc[:, 0].to_numpy(dtype=float)
    df, loc, scale = st.t.fit(x)

    out = pd.DataFrame({
        "ES Absolute": [es_t(df, loc, scale)],
        "ES Diff from Mean": [es_t(df, 0.0, scale)],
    })
    out.to_csv("testout8_5.csv", index=False)

if __name__ == "__main__":
    main()