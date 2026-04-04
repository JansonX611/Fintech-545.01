import numpy as np
import pandas as pd
from scipy.optimize import minimize

cov = pd.read_csv('test5_2.csv').values
risk_budget = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
target = risk_budget / risk_budget.sum()
n = cov.shape[0]


def portfolio_vol(w, cov_matrix):
    return np.sqrt(w @ cov_matrix @ w)


def risk_contribution(w, cov_matrix):
    sigma = portfolio_vol(w, cov_matrix)
    marginal = cov_matrix @ w
    return w * marginal / sigma


def objective(w, cov_matrix, target_budget):
    sigma = portfolio_vol(w, cov_matrix)
    rc = risk_contribution(w, cov_matrix)
    target_rc = sigma * target_budget
    return np.sum((rc - target_rc) ** 2)


x0 = np.ones(n) / n
bounds = [(0.0, 1.0)] * n
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

result = minimize(
    objective,
    x0,
    args=(cov, target),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'ftol': 1e-15, 'maxiter': 1000}
)

out = pd.DataFrame({'W': result.x})
out.to_csv('testout10_2.csv', index=False)
print(out)
