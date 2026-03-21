import pandas as pd
import numpy as np
import math

def american_discrete_div(opt, S0, K, T, r, vol, div_days, div_amts, total_days, steps=500):
    dt = T / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    div_steps = [round(day / total_days * steps) for day in div_days]
    div_map = dict(zip(div_steps, div_amts))

    j = np.arange(steps + 1)
    S = S0 * (u ** j) * (d ** (steps - j))
    vals = np.maximum(S - K, 0) if opt == "Call" else np.maximum(K - S, 0)

    for i in range(steps - 1, -1, -1):
        vals = disc * (p * vals[1:i+2] + (1 - p) * vals[:i+1])

        j = np.arange(i + 1)
        S = S0 * (u ** j) * (d ** (i - j))

        if i in div_map:
            D = div_map[i]
            vals = np.interp(S - D, S, vals, left=vals[0], right=vals[-1])

        ex = np.maximum(S - K, 0) if opt == "Call" else np.maximum(K - S, 0)
        vals = np.maximum(vals, ex)

    return float(vals[0])

df = pd.read_csv("test12_3.csv")

out = []
for _, row in df.iterrows():
    T = row["DaysToMaturity"] / row["DayPerYear"]
    div_days = [float(x) for x in str(row["DividendDates"]).split(",")]
    div_amts = [float(x) for x in str(row["DividendAmts"]).split(",")]

    value = american_discrete_div(
        row["Option Type"],
        row["Underlying"],
        row["Strike"],
        T,
        row["RiskFreeRate"],
        row["ImpliedVol"],
        div_days,
        div_amts,
        row["DaysToMaturity"],
        500
    )
    out.append([row["ID"], value])

pd.DataFrame(out, columns=["ID", "Value"]).to_csv("testout_12.3.csv", index=False)