# 📈 AlphaPulse — Investment Risk & Volatility Monitor

> A production-grade quantitative risk analytics engine for boutique investment firms.

---

## 🧭 Overview

AlphaPulse is a Python-powered financial analytics pipeline that computes critical market risk metrics and exports **Tableau-ready datasets** for interactive portfolio dashboards.

It calculates:
- ✅ **Value at Risk (VaR)** — Historical, Parametric, and Monte Carlo methods
- ✅ **Monte Carlo Simulation** — 10,000 GBM paths, 252-day forecast
- ✅ **Rolling Volatility** — 7 / 14 / 30 / 60-day annualised windows
- ✅ **Correlation Matrix** — Dynamic heatmap across all technical indicators
- ✅ **Core Price Metrics** — Candlestick data, Bollinger Bands, RSI, MACD, ATR

---

## 🗂️ Project Structure

```
alphapulse/
├── alphapulse_engine.py          ← Main analytics engine (run this)
├── cleaned_data.csv              ← Input: historical OHLCV + technical indicators
├── requirements.txt              ← Python dependencies
├── TABLEAU_DASHBOARD_GUIDE.md    ← Step-by-step Tableau build instructions
├── tableau_outputs/
│   ├── tableau_price_volume.csv       → Tab 1: Price, Volume, RSI, MACD
│   ├── tableau_rolling_volatility.csv → Tab 2: Rolling Volatility & Regimes
│   ├── tableau_monte_carlo.csv        → Tab 3: Monte Carlo Fan Chart
│   ├── tableau_var_metrics.csv        → Tab 4: VaR Executive Summary
│   └── tableau_correlation_matrix.csv → Tab 5: Correlation Heatmap
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/alphapulse.git
cd alphapulse
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the engine
```bash
python alphapulse_engine.py
```

All 5 Tableau CSVs will be regenerated inside `tableau_outputs/`.

---

## 📊 Tableau Dashboard Tabs

| Tab | CSV | Visuals |
|-----|-----|---------|
| 1 — Price & Volume | `tableau_price_volume.csv` | Candlestick, Volume Z-Score, RSI, MACD |
| 2 — Rolling Volatility | `tableau_rolling_volatility.csv` | Vol Fan, Regime Bars, Return Distribution |
| 3 — Monte Carlo | `tableau_monte_carlo.csv` | Cone of Uncertainty, What-If Parameter |
| 4 — VaR Summary | `tableau_var_metrics.csv` | KPI Tiles, Max Drawdown, VaR Table |
| 5 — Correlation | `tableau_correlation_matrix.csv` | Heatmap Matrix, Top Pairs Bar |

See [`TABLEAU_DASHBOARD_GUIDE.md`](./TABLEAU_DASHBOARD_GUIDE.md) for detailed sheet-by-sheet build instructions.

---

## 📐 Risk Metrics (Sample Output)

| Metric | Value |
|--------|-------|
| Portfolio Value | $1,000,000 |
| Historical VaR 95% (1-day) | −2.78% → **$27,812** |
| CVaR / Expected Shortfall | −4.55% → **$45,541** |
| Max Drawdown | **−40.30%** (Sept 2022) |
| Monte Carlo Median (1-yr) | **$304.90** |
| Monte Carlo Bear Case (P5) | $155.53 |
| Monte Carlo Bull Case (P95) | $595.34 |

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Data Ingestion | `pandas` |
| Quantitative Math | `numpy`, `scipy` |
| Simulation | NumPy vectorised GBM (10,000 paths) |
| Visualisation | Tableau (connect CSVs directly) |
| Scheduling | cron / Windows Task Scheduler |

---

## 🔁 Automating Data Refresh

To refresh daily after market close, schedule `alphapulse_engine.py`:

**Linux / macOS (cron):**
```bash
30 18 * * 1-5 cd /path/to/alphapulse && python alphapulse_engine.py
```

**Windows Task Scheduler:**
- Action: `python C:\path\to\alphapulse\alphapulse_engine.py`
- Trigger: Daily, 6:30 PM Mon–Fri

Tableau's **Hyper API** or **Tableau Prep** can then auto-ingest the refreshed CSVs without any manual steps.

---

## 📅 Development Roadmap

| Week | Focus | Status |
|------|-------|--------|
| Week 5 | Data Acquisition & Cleaning | ✅ Done |
| Week 6 | Quantitative Analysis (VaR, Monte Carlo) | ✅ Done |
| Week 7 | Tableau Visual Storytelling + What-If Params | ✅ Done |
| Week 8 | Automation + Executive Summary Tab | ✅ Done |

---




