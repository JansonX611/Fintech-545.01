import numpy as np
import pandas as pd

factor_returns = pd.read_csv('test11_2_factor_returns.csv')
stock_returns = pd.read_csv('test11_2_stock_returns.csv')
beta = pd.read_csv('test11_2_beta.csv').set_index('Stock')
weights = pd.read_csv('test11_2_weights.csv')['W'].values

beta_matrix = beta.loc[stock_returns.columns].values

stock_values = np.zeros((len(stock_returns) + 1, len(weights)))
stock_values[0] = weights

for t in range(len(stock_returns)):
    stock_values[t + 1] = stock_values[t] * (1 + stock_returns.iloc[t].values)

portfolio_values = stock_values.sum(axis=1)
current_weights = stock_values[:-1] / portfolio_values[:-1, None]

factor_exposure = current_weights @ beta_matrix
factor_contribution = factor_returns.values * factor_exposure

stock_contribution = current_weights * stock_returns.values
portfolio_return = stock_contribution.sum(axis=1)
alpha_contribution = portfolio_return - factor_contribution.sum(axis=1)
portfolio_total_return = (1 + portfolio_return).prod() - 1

factor_total_return = (1 + factor_returns).prod() - 1
alpha_total_return = (1 + alpha_contribution).prod() - 1

k = np.log(1 + portfolio_total_return) / portfolio_total_return
kt = np.where(
    np.abs(portfolio_return) > 1e-15,
    np.log(1 + portfolio_return) / portfolio_return,
    1 / (1 + portfolio_return)
)
carino = kt / k

return_attribution = (factor_contribution * carino[:, None]).sum(axis=0)
alpha_return_attribution = (alpha_contribution * carino).sum()

all_contribution = np.column_stack([factor_contribution, alpha_contribution])
cov_matrix = np.cov(all_contribution, rowvar=False, ddof=1)
portfolio_vol = np.std(portfolio_return, ddof=1)
vol_attribution = cov_matrix.sum(axis=1) / portfolio_vol

out = pd.DataFrame(
    [
        list(factor_total_return.values) + [alpha_total_return, portfolio_total_return],
        list(return_attribution) + [alpha_return_attribution, portfolio_total_return],
        list(vol_attribution) + [portfolio_vol]
    ],
    index=['TotalReturn', 'Return Attribution', 'Vol Attribution'],
    columns=list(factor_returns.columns) + ['Alpha', 'Portfolio']
)

out.index.name = 'Value'
out.to_csv('testout11_2.csv')
print(out)
