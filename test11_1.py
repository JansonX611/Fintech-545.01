import numpy as np
import pandas as pd

returns = pd.read_csv('test11_1_returns.csv')
weights = pd.read_csv('test11_1_weights.csv')['W'].values

asset_total_return = (1 + returns).prod() - 1

asset_values = np.zeros((len(returns) + 1, len(weights)))
asset_values[0] = weights

for t in range(len(returns)):
    asset_values[t + 1] = asset_values[t] * (1 + returns.iloc[t].values)

portfolio_values = asset_values.sum(axis=1)
current_weights = asset_values[:-1] / portfolio_values[:-1, None]
period_contribution = current_weights * returns.values
portfolio_return = period_contribution.sum(axis=1)
portfolio_total_return = (1 + portfolio_return).prod() - 1

k = np.log(1 + portfolio_total_return) / portfolio_total_return
kt = np.where(
    np.abs(portfolio_return) > 1e-15,
    np.log(1 + portfolio_return) / portfolio_return,
    1 / (1 + portfolio_return)
)
carino = kt / k

return_attribution = (period_contribution * carino[:, None]).sum(axis=0)

cov_matrix = np.cov(period_contribution, rowvar=False, ddof=1)
portfolio_vol = np.std(portfolio_return, ddof=1)
vol_attribution = cov_matrix.sum(axis=1) / portfolio_vol

out = pd.DataFrame(
    [
        list(asset_total_return.values) + [portfolio_total_return],
        list(return_attribution) + [portfolio_total_return],
        list(vol_attribution) + [portfolio_vol]
    ],
    index=['TotalReturn', 'Return Attribution', 'Vol Attribution'],
    columns=list(returns.columns) + ['Portfolio']
)

out.index.name = 'Value'
out.to_csv('testout11_1.csv')
print(out)
