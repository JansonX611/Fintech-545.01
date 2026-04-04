import numpy as np
import pandas as pd
from scipy.optimize import minimize

cov = pd.read_csv('test5_2.csv').values
mu = pd.read_csv('test10_3_means.csv')['Mean'].values
rf = 0.04
n = len(mu)


def neg_sharpe(w, mean_returns, cov_matrix, risk_free_rate):
    port_return = w @ mean_returns
    port_vol = np.sqrt(w @ cov_matrix @ w)
    return -((port_return - risk_free_rate) / port_vol)


x0 = np.ones(n) / n
bounds = [(0.1, 0.5)] * n
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

result = minimize(
    neg_sharpe,
    x0,
    args=(mu, cov, rf),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'ftol': 1e-15, 'maxiter': 1000}
)

out = pd.DataFrame({'W': result.x})
out.to_csv('testout10_4.csv', index=False)
print(out)
