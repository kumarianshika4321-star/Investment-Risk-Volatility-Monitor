# AlphaPulse — Tableau Dashboard Blueprint
## Zaalima Development Pvt. Ltd. | Confidential

---

## FILES TO CONNECT IN TABLEAU

| CSV File | Dashboard Tab |
|---|---|
| `tableau_price_volume.csv` | Tab 1 — Price & Volume |
| `tableau_rolling_volatility.csv` | Tab 2 — Rolling Volatility |
| `tableau_monte_carlo.csv` | Tab 3 — Monte Carlo |
| `tableau_var_metrics.csv` | Tab 4 — VaR Summary |
| `tableau_correlation_matrix.csv` | Tab 5 — Correlation Heatmap |

---

## TAB 1: PRICE & VOLUME DASHBOARD

### Sheet 1A — Candlestick / Line Chart
- **Data source**: `tableau_price_volume.csv`
- **X-axis**: `Date` (continuous)
- **Y-axis**: `Close`
- **Color**: `Candle_Dir` (Up = green, Down = red)
- **Add reference lines** for `SMA_30`, `SMA_200`, `BB_Upper`, `BB_Lower`
- **Tooltip**: Open, High, Low, Close, Volume, Daily_Return

### Sheet 1B — Volume Bar Chart
- **X-axis**: `Date`
- **Y-axis**: `Volume`
- **Color**: `Volume_ZScore` (diverging color — blue = low, red = high)
- Place directly below Sheet 1A (synchronized x-axis)

### Sheet 1C — RSI Indicator
- **X-axis**: `Date`, **Y-axis**: `RSI`
- Add **constant reference lines** at 70 (overbought) and 30 (oversold)
- Color: conditional — red above 70, green below 30

### Sheet 1D — MACD Panel
- Dual-axis: `MACD` (line) + `Signal_Line` (line) + `MACD_Histogram` (bar)
- Color histogram bars: green if positive, red if negative

### Dashboard Assembly (Tab 1)
- Stack: 1A (top, 60%) → 1B (20%) → 1C (10%) → 1D (10%)
- Add **Date Range filter** (slider) that controls all 4 sheets simultaneously
- Add **Momentum KPI tiles**: Momentum_7d and Momentum_30d

---

## TAB 2: ROLLING VOLATILITY

### Sheet 2A — Volatility Fan Chart
- **Data source**: `tableau_rolling_volatility.csv`
- Plot all four rolling windows: 7d, 14d, 30d, 60d as separate lines
- Use a **parameter** to toggle between annualised vol windows
- Background color bands: Low / Medium / High regime zones

### Sheet 2B — Volatility Regime Bar
- **Color**: `Vol_Regime` (green / yellow / red)
- Stacked bar showing % of time spent in each regime per year
  - Use `YEAR(Date)` on columns, `COUNTD(Date)` on rows, colored by `Vol_Regime`

### Sheet 2C — Return Distribution Histogram
- `Daily_Return` on columns (binned, bin size = 0.5%)
- Count on rows — this shows the return distribution shape
- Add a **normal curve reference** as annotation

---

## TAB 3: MONTE CARLO SIMULATION

### Sheet 3A — Fan / Cone of Uncertainty
- **Data source**: `tableau_monte_carlo.csv`
- **X-axis**: `Forecast_Date`
- Plot these as separate lines:
  - `P5_Price` (dashed, red) — bear case
  - `P25_Price` (dotted, orange)
  - `Median_Price` (solid, white/blue, thick)
  - `P75_Price` (dotted, light green)
  - `P95_Price` (dashed, green) — bull case
- Use **Band shading**: shade between P5 and P95 (light gray)
- Add a **vertical reference line** at Day=0 with label "Today: $100.93"

### Sheet 3B — Terminal Distribution (Day 252)
- Filter to `Day = 252`
- Show distribution of: P5, P25, Median, Mean, P75, P95 as a horizontal bar
- KPI tiles: Return_P5_Pct, Return_Mean_Pct, Return_P95_Pct

### "What-If" Parameter (Tableau Parameter Control)
- Create parameter: **"Shock Factor %"** (range −50% to +50%, step 1%)
- Create calculated field: `[Mean_Price] * (1 + [Shock Factor %] / 100)`
- This instantly re-prices all Monte Carlo lines — fulfills the
  "What if Tech drops 10%?" requirement from the brief

---

## TAB 4: VAR & RISK METRICS (Executive Summary)

### Sheet 4A — VaR Comparison Table
- **Data source**: `tableau_var_metrics.csv`
- Rows: `Method` (Historical / Parametric / Monte Carlo)
- Columns: `Confidence_Label` (90%, 95%, 99%)
- Values: `VaR_1D_Dollar` (formatted as currency)
- Color: diverging — deeper red = larger loss

### Sheet 4B — VaR Bar Chart
- Grouped bar: `Method` on X, `VaR_1D_Dollar` on Y
- Color by `Confidence_Label`

### Sheet 4C — KPI Tiles (Text / Number sheets)
- **Current VaR 95% (1D)**: $27,812
- **CVaR / Expected Shortfall**: $45,541
- **Max Drawdown**: −40.30%
- **Max Drawdown Date**: 2022-09-01
- **Portfolio Value**: $1,000,000

### Sheet 4D — Drawdown Chart
- Connect to `tableau_price_volume.csv`
- Create calculated field:
  `RUNNING_MAX([Close])` → then `([Close] - RUNNING_MAX([Close])) / RUNNING_MAX([Close]) * 100`
- Area chart, filled red below zero

---

## TAB 5: CORRELATION HEATMAP

### Sheet 5A — Heatmap Matrix
- **Data source**: `tableau_correlation_matrix.csv`
- Rows: `Indicator_X`, Columns: `Indicator_Y`
- Color: **Diverging** (Red = −1 strong negative, White = 0, Blue = +1 strong positive)
- **Mark**: Square
- **Label**: `Correlation_Label`
- **Size**: `Abs_Correlation` (larger square = stronger correlation)
- **Filter**: `Strength` (allow user to filter by "Strong Positive" etc.)

### Sheet 5B — Top Correlations Bar
- Sorted bar chart of `Abs_Correlation` filtered to `Indicator_X != Indicator_Y`
- Show top 10 pairs

---

## TABLEAU DASHBOARD LAYOUT GUIDE

```
┌─────────────────────────────────────────────────────────────┐
│  AlphaPulse  |  Investment Risk & Volatility Monitor        │
│  [Date Range Filter]  [What-If Shock %]  [Volatility Window]│
├──────────────┬──────────────────────────────────────────────┤
│ TAB 1        │ TAB 2         │ TAB 3        │ TAB 4  │ TAB 5│
│ Price/Volume │ Volatility    │ Monte Carlo  │ VaR    │ Corr │
├──────────────┴──────────────────────────────────────────────┤
│  [KPI Row: Last Close | VaR 95% | Max Drawdown | Vol 30d]   │
└─────────────────────────────────────────────────────────────┘
```

---

## TABLEAU TIPS

1. **Blend all CSVs on `Date`** where applicable (Relationships panel)
2. Use **Dashboard Actions** — clicking a date on Tab 1 filters the Vol tab
3. Set **Date** field to "Exact Date" not "Year" for granular drill-down
4. For the Monte Carlo cone — use **Dual Axis** with synchronized axes
5. Export each tab as a **Tableau Story** for the Executive Summary presentation

---

## DATA REFRESH AUTOMATION

Run `alphapulse_engine.py` on a schedule (e.g., Windows Task Scheduler / cron):
```
# Daily at 6:30 AM after market open
0 6 * * 1-5 cd /path/to/alphapulse && python alphapulse_engine.py
```
Tableau's **Hyper API** or **Tableau Prep** can then auto-ingest the refreshed CSVs.
