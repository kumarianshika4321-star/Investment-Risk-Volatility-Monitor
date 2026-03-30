"""
╔══════════════════════════════════════════════════════════╗
║           AlphaPulse — Investment Risk & Volatility      ║
║           Analytics Engine  |  Zaalima Development       ║
╚══════════════════════════════════════════════════════════╝

Outputs five Tableau-ready CSVs:
  1. tableau_price_volume.csv         → Price / Volume / Returns dashboard
  2. tableau_rolling_volatility.csv   → 30-day rolling volatility
  3. tableau_monte_carlo.csv          → Monte Carlo simulation (10,000 runs)
  4. tableau_var_metrics.csv          → Value at Risk (VaR) summary
  5. tableau_correlation_matrix.csv   → Correlation heatmap (technical indicators)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────
INPUT_FILE   = "cleaned_data.csv"
OUTPUT_DIR   = "tableau_outputs"
N_SIMULATIONS = 10_000
FORECAST_DAYS = 252          # ~1 trading year ahead
VAR_CONFIDENCE = [0.90, 0.95, 0.99]
ROLLING_VOL_WINDOW = 30
RANDOM_SEED = 42

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────
#  LOAD & VALIDATE
# ─────────────────────────────────────────────────────────
print("▶  Loading cleaned_data.csv …")
df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"   {len(df):,} rows  |  {df['Date'].min().date()} → {df['Date'].max().date()}")
assert df.isnull().sum().sum() == 0, "Null values detected — re-run cleaning step."

# ─────────────────────────────────────────────────────────
#  1. PRICE / VOLUME / RETURNS  (core metrics)
# ─────────────────────────────────────────────────────────
print("\n▶  Building Price / Volume / Returns table …")

price_vol = df[[
    "Date", "Open", "High", "Low", "Close", "Adj Close",
    "Volume", "Daily_Return",
    "SMA_7", "SMA_14", "SMA_30", "SMA_200",
    "EMA_12", "EMA_26",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "RSI", "MACD", "Signal_Line", "MACD_Histogram",
    "ATR"
]].copy()

# Cumulative return from start
price_vol["Cumulative_Return_Pct"] = (
    (price_vol["Close"] / price_vol["Close"].iloc[0] - 1) * 100
).round(4)

# 7-day & 30-day momentum
price_vol["Momentum_7d"]  = price_vol["Close"].pct_change(7)  * 100
price_vol["Momentum_30d"] = price_vol["Close"].pct_change(30) * 100

# Volume z-score vs 30-day average
vol_mean = price_vol["Volume"].rolling(30).mean()
vol_std  = price_vol["Volume"].rolling(30).std()
price_vol["Volume_ZScore"] = ((price_vol["Volume"] - vol_mean) / vol_std).round(4)

# Candle body size (useful for Tableau candlestick)
price_vol["Candle_Body"]   = (price_vol["Close"] - price_vol["Open"]).abs().round(4)
price_vol["Candle_Dir"]    = np.where(price_vol["Close"] >= price_vol["Open"], "Up", "Down")

price_vol.to_csv(f"{OUTPUT_DIR}/tableau_price_volume.csv", index=False)
print(f"   ✓  tableau_price_volume.csv  ({len(price_vol)} rows)")

# ─────────────────────────────────────────────────────────
#  2. ROLLING VOLATILITY  (30-day)
# ─────────────────────────────────────────────────────────
print("\n▶  Calculating rolling volatility …")

vol_df = df[["Date", "Close", "Daily_Return"]].copy()

# Annualised rolling std of daily log returns
log_ret = np.log(df["Close"] / df["Close"].shift(1))

for window in [7, 14, 30, 60]:
    rolling_vol = log_ret.rolling(window).std() * np.sqrt(252) * 100
    vol_df[f"RollingVol_{window}d_Ann_Pct"] = rolling_vol.round(4)

# Realised variance (30-day)
vol_df["RealisedVariance_30d"] = (log_ret.rolling(30).var() * 252).round(6)

# Volatility regime label (low / medium / high based on 30-day ann vol)
v30 = vol_df["RollingVol_30d_Ann_Pct"]
vol_df["Vol_Regime"] = pd.cut(
    v30,
    bins=[-np.inf, 20, 40, np.inf],
    labels=["Low (<20%)", "Medium (20-40%)", "High (>40%)"]
)

vol_df.to_csv(f"{OUTPUT_DIR}/tableau_rolling_volatility.csv", index=False)
print(f"   ✓  tableau_rolling_volatility.csv  ({len(vol_df)} rows)")

# ─────────────────────────────────────────────────────────
#  3. MONTE CARLO SIMULATION  (10,000 paths, 252 days)
# ─────────────────────────────────────────────────────────
print(f"\n▶  Running Monte Carlo ({N_SIMULATIONS:,} simulations × {FORECAST_DAYS} days) …")

# Use last 252 days for parameter estimation (recent market regime)
recent_returns = log_ret.dropna().tail(252)
mu    = recent_returns.mean()       # daily drift
sigma = recent_returns.std()        # daily volatility
S0    = df["Close"].iloc[-1]        # last known price
last_date = df["Date"].iloc[-1]

print(f"   Parameters → μ={mu:.6f}  σ={sigma:.6f}  S₀={S0:.4f}")

# GBM: S(t) = S0 * exp( (μ - σ²/2)*t + σ*√t*Z )
dt = 1  # 1 trading day
Z  = np.random.standard_normal((FORECAST_DAYS, N_SIMULATIONS))
daily_shocks = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
price_paths  = S0 * np.cumprod(daily_shocks, axis=0)  # shape (252, 10000)

# ── Summary statistics per simulation day ──
sim_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)

mc_summary = pd.DataFrame({
    "Forecast_Date":   sim_dates,
    "Day":             np.arange(1, FORECAST_DAYS + 1),
    "Mean_Price":      price_paths.mean(axis=1).round(4),
    "Median_Price":    np.median(price_paths, axis=1).round(4),
    "P5_Price":        np.percentile(price_paths, 5,  axis=1).round(4),
    "P10_Price":       np.percentile(price_paths, 10, axis=1).round(4),
    "P25_Price":       np.percentile(price_paths, 25, axis=1).round(4),
    "P75_Price":       np.percentile(price_paths, 75, axis=1).round(4),
    "P90_Price":       np.percentile(price_paths, 90, axis=1).round(4),
    "P95_Price":       np.percentile(price_paths, 95, axis=1).round(4),
    "Std_Price":       price_paths.std(axis=1).round(4),
    "Min_Price":       price_paths.min(axis=1).round(4),
    "Max_Price":       price_paths.max(axis=1).round(4),
    "S0_Reference":    S0,
})

mc_summary["Return_Mean_Pct"]   = ((mc_summary["Mean_Price"]   / S0 - 1) * 100).round(4)
mc_summary["Return_P5_Pct"]     = ((mc_summary["P5_Price"]     / S0 - 1) * 100).round(4)
mc_summary["Return_P95_Pct"]    = ((mc_summary["P95_Price"]    / S0 - 1) * 100).round(4)

mc_summary.to_csv(f"{OUTPUT_DIR}/tableau_monte_carlo.csv", index=False)
print(f"   ✓  tableau_monte_carlo.csv  ({len(mc_summary)} rows, {N_SIMULATIONS:,} paths)")

# ─────────────────────────────────────────────────────────
#  4. VALUE AT RISK (VaR) METRICS
# ─────────────────────────────────────────────────────────
print("\n▶  Computing Value at Risk …")

daily_returns_pct = df["Daily_Return"].dropna() / 100  # convert to decimals

portfolio_value = 1_000_000  # assume $1M portfolio for VaR dollar amounts

var_records = []

for conf in VAR_CONFIDENCE:
    # ── Historical VaR
    hist_var_pct = np.percentile(daily_returns_pct, (1 - conf) * 100)
    hist_var_dollar = abs(hist_var_pct * portfolio_value)

    # ── Parametric VaR (Normal distribution)
    from scipy.stats import norm
    param_var_pct = norm.ppf(1 - conf, loc=daily_returns_pct.mean(), scale=daily_returns_pct.std())
    param_var_dollar = abs(param_var_pct * portfolio_value)

    # ── Monte Carlo VaR (from 1-day simulated returns)
    mc_1day_returns = (price_paths[0] / S0) - 1  # day-1 returns across all sims
    mc_var_pct = np.percentile(mc_1day_returns, (1 - conf) * 100)
    mc_var_dollar = abs(mc_var_pct * portfolio_value)

    # ── Expected Shortfall (CVaR)
    es_threshold = np.percentile(daily_returns_pct, (1 - conf) * 100)
    cvar_pct = daily_returns_pct[daily_returns_pct <= es_threshold].mean()
    cvar_dollar = abs(cvar_pct * portfolio_value)

    # ── 10-day scaled VaR (Basel square-root-of-time rule)
    hist_var_10d_pct    = hist_var_pct    * np.sqrt(10)
    hist_var_10d_dollar = hist_var_dollar * np.sqrt(10)

    var_records.append({
        "Confidence_Level":        conf,
        "Confidence_Label":        f"{int(conf*100)}%",
        "Method":                  "Historical",
        "VaR_1D_Pct":              round(hist_var_pct * 100, 4),
        "VaR_1D_Dollar":           round(hist_var_dollar, 2),
        "VaR_10D_Pct":             round(hist_var_10d_pct * 100, 4),
        "VaR_10D_Dollar":          round(hist_var_10d_dollar, 2),
        "CVaR_1D_Pct":             round(cvar_pct * 100, 4),
        "CVaR_1D_Dollar":          round(cvar_dollar, 2),
        "Portfolio_Value":         portfolio_value,
    })
    var_records.append({
        "Confidence_Level":        conf,
        "Confidence_Label":        f"{int(conf*100)}%",
        "Method":                  "Parametric",
        "VaR_1D_Pct":              round(param_var_pct * 100, 4),
        "VaR_1D_Dollar":           round(param_var_dollar, 2),
        "VaR_10D_Pct":             round(param_var_pct * 100 * np.sqrt(10), 4),
        "VaR_10D_Dollar":          round(param_var_dollar * np.sqrt(10), 2),
        "CVaR_1D_Pct":             round(cvar_pct * 100, 4),
        "CVaR_1D_Dollar":          round(cvar_dollar, 2),
        "Portfolio_Value":         portfolio_value,
    })
    var_records.append({
        "Confidence_Level":        conf,
        "Confidence_Label":        f"{int(conf*100)}%",
        "Method":                  "Monte Carlo",
        "VaR_1D_Pct":              round(mc_var_pct * 100, 4),
        "VaR_1D_Dollar":           round(mc_var_dollar, 2),
        "VaR_10D_Pct":             round(mc_var_pct * 100 * np.sqrt(10), 4),
        "VaR_10D_Dollar":          round(mc_var_dollar * np.sqrt(10), 2),
        "CVaR_1D_Pct":             round(cvar_pct * 100, 4),
        "CVaR_1D_Dollar":          round(cvar_dollar, 2),
        "Portfolio_Value":         portfolio_value,
    })

var_df = pd.DataFrame(var_records)

# Add Max Drawdown
rolling_max = df["Close"].cummax()
drawdown    = (df["Close"] - rolling_max) / rolling_max * 100
max_dd_pct  = drawdown.min()
max_dd_date = df["Date"].iloc[drawdown.idxmin()]

# Add a summary row
summary_row = {
    "Confidence_Level": None, "Confidence_Label": "Summary",
    "Method": "Max Drawdown",
    "VaR_1D_Pct": round(max_dd_pct, 4),
    "VaR_1D_Dollar": round(abs(max_dd_pct / 100) * portfolio_value, 2),
    "VaR_10D_Pct": None, "VaR_10D_Dollar": None,
    "CVaR_1D_Pct": None, "CVaR_1D_Dollar": None,
    "Portfolio_Value": portfolio_value,
}
var_df = pd.concat([var_df, pd.DataFrame([summary_row])], ignore_index=True)

var_df.to_csv(f"{OUTPUT_DIR}/tableau_var_metrics.csv", index=False)
print(f"   ✓  tableau_var_metrics.csv  ({len(var_df)} rows)")
print(f"   Max Drawdown: {max_dd_pct:.2f}%  on {max_dd_date.date()}")

# ─────────────────────────────────────────────────────────
#  5. CORRELATION MATRIX  (technical indicators)
# ─────────────────────────────────────────────────────────
print("\n▶  Building correlation matrix …")

corr_cols = [
    "Close", "Daily_Return", "Volume",
    "SMA_7", "SMA_30", "SMA_200",
    "RSI", "MACD", "ATR",
    "BB_Upper", "BB_Lower",
    "Volatility"
]

corr_matrix = df[corr_cols].corr().round(4)

# Melt to long form for Tableau heatmap
corr_long = corr_matrix.reset_index().melt(id_vars="index")
corr_long.columns = ["Indicator_X", "Indicator_Y", "Correlation"]
corr_long["Abs_Correlation"] = corr_long["Correlation"].abs()
corr_long["Correlation_Label"] = corr_long["Correlation"].map(lambda x: f"{x:.4f}")

# Correlation strength label
bins   = [-1.01, -0.7, -0.3, 0.3, 0.7, 1.01]
labels = ["Strong Negative", "Weak Negative", "Neutral", "Weak Positive", "Strong Positive"]
corr_long["Strength"] = pd.cut(corr_long["Correlation"], bins=bins, labels=labels)

corr_long.to_csv(f"{OUTPUT_DIR}/tableau_correlation_matrix.csv", index=False)
print(f"   ✓  tableau_correlation_matrix.csv  ({len(corr_long)} rows)")

# ─────────────────────────────────────────────────────────
#  PRINT EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  AlphaPulse — EXECUTIVE RISK SUMMARY")
print("═" * 60)
print(f"  Portfolio Value          : ${portfolio_value:,.0f}")
print(f"  Data Period              : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Last Close               : ${S0:.4f}")
print()
hist_var95 = var_df[(var_df["Confidence_Label"]=="95%") & (var_df["Method"]=="Historical")].iloc[0]
print(f"  Historical VaR (95%, 1D) : {hist_var95['VaR_1D_Pct']:.2f}%  |  ${hist_var95['VaR_1D_Dollar']:,.0f}")
print(f"  CVaR / Expected Shortfall: {hist_var95['CVaR_1D_Pct']:.2f}%  |  ${hist_var95['CVaR_1D_Dollar']:,.0f}")
print(f"  Max Drawdown             : {max_dd_pct:.2f}%  (peak on {max_dd_date.date()})")
print()
mc_1yr = mc_summary.iloc[-1]
print(f"  Monte Carlo 1-Year Forecast (median)  : ${mc_1yr['Median_Price']:.2f}")
print(f"  5th Pct (bear case)                   : ${mc_1yr['P5_Price']:.2f}")
print(f"  95th Pct (bull case)                  : ${mc_1yr['P95_Price']:.2f}")
print("═" * 60)
print(f"\n✅  All 5 Tableau CSVs saved to  ./{OUTPUT_DIR}/\n")
