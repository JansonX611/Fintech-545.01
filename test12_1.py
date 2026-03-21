import pandas as pd
import math

def N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def n(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def gbsm_greeks(opt, S, K, days, day_year, r, q, vol):
    T = days / day_year
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)

    if opt == "Call":
        value = S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
        delta = math.exp(-q * T) * N(d1)
        rho = K * T * math.exp(-r * T) * N(d2)
        theta = (-S * math.exp(-q * T) * n(d1) * vol / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * N(d2)
                 + q * S * math.exp(-q * T) * N(d1))
    else:
        value = K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)
        delta = math.exp(-q * T) * (N(d1) - 1)
        rho = -K * T * math.exp(-r * T) * N(-d2)
        theta = (-S * math.exp(-q * T) * n(d1) * vol / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * N(-d2)
                 - q * S * math.exp(-q * T) * N(-d1))

    gamma = math.exp(-q * T) * n(d1) / (S * vol * math.sqrt(T))
    vega = S * math.exp(-q * T) * n(d1) * math.sqrt(T)

    return value, delta, gamma, vega, rho, theta

df = pd.read_csv("test12_1.csv").dropna()

out = []
for _, row in df.iterrows():
    value, delta, gamma, vega, rho, theta = gbsm_greeks(
        row["Option Type"],
        row["Underlying"],
        row["Strike"],
        row["DaysToMaturity"],
        row["DayPerYear"],
        row["RiskFreeRate"],
        row["DividendRate"],
        row["ImpliedVol"]
    )
    out.append([row["ID"], value, delta, gamma, vega, rho, theta])

out_df = pd.DataFrame(out, columns=["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"])
out_df.to_csv("testout_12.1.csv", index=False)
print(out_df)