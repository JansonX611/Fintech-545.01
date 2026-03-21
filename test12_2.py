import pandas as pd
import numpy as np
import math

def american(opt, S, K, T, r, q, vol, steps=500):
    dt = T / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1)
    ST = S * (u ** j) * (d ** (steps - j))
    vals = np.maximum(ST - K, 0) if opt == "Call" else np.maximum(K - ST, 0)

    for i in range(steps - 1, -1, -1):
        vals = disc * (p * vals[1:i+2] + (1 - p) * vals[:i+1])
        j = np.arange(i + 1)
        ST = S * (u ** j) * (d ** (i - j))
        ex = np.maximum(ST - K, 0) if opt == "Call" else np.maximum(K - ST, 0)
        vals = np.maximum(vals, ex)

    return float(vals[0])

def greeks(opt, S, K, T, r, q, vol):
    V = american(opt, S, K, T, r, q, vol, 500)

    dS = 1e-3
    dG = 1.5
    dV = 1e-5
    dR = 4.701851489299857e-06
    dT = 2.0022003718155845e-07

    delta = (american(opt, S + dS, K, T, r, q, vol, 500) -
             american(opt, S - dS, K, T, r, q, vol, 500)) / (2 * dS)

    gamma = (american(opt, S + dG, K, T, r, q, vol, 500) -
             2 * V +
             american(opt, S - dG, K, T, r, q, vol, 500)) / (dG ** 2)

    vega = (american(opt, S, K, T, r, q, vol + dV, 500) -
            american(opt, S, K, T, r, q, vol - dV, 500)) / (2 * dV)

    rho = (american(opt, S, K, T, r + dR, q + dR, vol, 500) -
           american(opt, S, K, T, r - dR, q - dR, vol, 500)) / (2 * dR)

    theta = (V - american(opt, S, K, T - dT, r, q, vol, 500)) / dT

    return V, delta, gamma, vega, rho, theta

df = pd.read_csv("test12_1.csv").dropna()

out = []
for _, row in df.iterrows():
    T = row["DaysToMaturity"] / row["DayPerYear"]
    value, delta, gamma, vega, rho, theta = greeks(
        row["Option Type"],
        row["Underlying"],
        row["Strike"],
        T,
        row["RiskFreeRate"],
        row["DividendRate"],
        row["ImpliedVol"]
    )
    out.append([row["ID"], value, delta, gamma, vega, rho, theta])

pd.DataFrame(out, columns=["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"])\
  .to_csv("testout_12.2.csv", index=False)