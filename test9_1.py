import pandas as pd
import numpy as np
import scipy.stats as st

ALPHA = 0.05
N_SIM = 100000

def spearman_corr_matrix(U: np.ndarray) -> np.ndarray:
    k = U.shape[1]
    corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            r, _ = st.spearmanr(U[:, i], U[:, j])
            corr[i, j] = corr[j, i] = r
    return corr

def simulate_pca_corr(corr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, 0.0, None)
    L = vecs @ np.diag(np.sqrt(vals))
    z = rng.standard_normal((n, corr.shape[0]))
    return z @ L.T

def var_es(loss: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    q = np.quantile(loss, alpha)
    tail = loss[loss <= q]
    return -float(q), -float(np.mean(tail))

def main():
    portfolio = pd.read_csv("test9_1_portfolio.csv")
    returns = pd.read_csv("test9_1_returns.csv")

    stocks = portfolio["Stock"].astype(str).tolist()

    models = {}
    u_cols = []

    for stock, dist in zip(stocks, portfolio["Distribution"].astype(str)):
        x = returns[stock].to_numpy(dtype=float)
        d = dist.strip().lower()

        if d.startswith("n"):
            mu = float(np.mean(x))
            sigma = float(np.std(x, ddof=1))
            models[stock] = ("norm", (mu, sigma))
            u = st.norm.cdf(x, loc=mu, scale=sigma)
        else:
            df, loc, scale = st.t.fit(x)
            models[stock] = ("t", (float(df), float(loc), float(scale)))
            u = st.t.cdf(x, df, loc=loc, scale=scale)

        u_cols.append(u)

    U = np.column_stack(u_cols)
    spcor = spearman_corr_matrix(U)

    rng = np.random.default_rng(2)
    z_sim = simulate_pca_corr(spcor, N_SIM, rng)
    u_sim = st.norm.cdf(z_sim)

    sim_ret = np.zeros_like(u_sim)
    for j, stock in enumerate(stocks):
        kind, par = models[stock]
        if kind == "norm":
            mu, sigma = par
            sim_ret[:, j] = st.norm.ppf(u_sim[:, j], loc=mu, scale=sigma)
        else:
            df, loc, scale = par
            sim_ret[:, j] = st.t.ppf(u_sim[:, j], df, loc=loc, scale=scale)

    current_value = portfolio["Holding"].to_numpy(dtype=float) * portfolio["Starting Price"].to_numpy(dtype=float)
    pnl = sim_ret * current_value

    rows = []
    for j, stock in enumerate(stocks):
        v, e = var_es(pnl[:, j], ALPHA)
        cv = float(current_value[j])
        rows.append([stock, v, e, v / cv, e / cv])

    total_pnl = np.sum(pnl, axis=1)
    total_cv = float(np.sum(current_value))
    v, e = var_es(total_pnl, ALPHA)
    rows.append(["Total", v, e, v / total_cv, e / total_cv])

    out = pd.DataFrame(rows, columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])
    out.to_csv("testout9_1.csv", index=False)

if __name__ == "__main__":
    main()