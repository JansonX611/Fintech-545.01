import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t

df = pd.read_csv("test7_3.csv")
y = df["y"].to_numpy()
X = df.drop(columns=["y"]).to_numpy()

n, k = X.shape
X1 = np.column_stack([np.ones(n), X])

beta_ols, *_ = np.linalg.lstsq(X1, y, rcond=None)
alpha0 = beta_ols[0]
b0 = beta_ols[1:]

e0 = y - (alpha0 + X @ b0)
mu0 = e0.mean()
sigma0 = max(e0.std(ddof=1), 1e-8)
nu0 = 10.0

p0 = np.concatenate([[alpha0], b0, [mu0], [np.log(sigma0)], [np.log(max(nu0 - 2.0, 1e-6))]])

def unpack(p):
    alpha = p[0]
    b = p[1:1 + k]
    mu = p[1 + k]
    sigma = np.exp(p[2 + k])
    nu = 2.0 + np.exp(p[3 + k])
    return alpha, b, mu, sigma, nu

def nll(p):
    alpha, b, mu, sigma, nu = unpack(p)
    e = y - (alpha + X @ b)
    z = (e - mu) / sigma
    return -np.sum(t.logpdf(z, df=nu) - np.log(sigma))

res = minimize(nll, p0, method="L-BFGS-B", options={"maxiter": 20000})

alpha, b, mu, sigma, nu = unpack(res.x)

alpha = alpha + mu
mu = 0.0

out = {"mu": [mu], "sigma": [sigma], "nu": [nu], "Alpha": [alpha]}
for i in range(k):
    out[f"B{i+1}"] = [b[i]]

pd.DataFrame(out).to_csv("testout7_3.csv", index=False)
