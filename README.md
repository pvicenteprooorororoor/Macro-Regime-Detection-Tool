# Macro Regime Detection Tool

A quantitative tool for detecting macroeconomic regimes using **K-Means Clustering** and **Hidden Markov Models (HMM)**, with tactical asset allocation recommendations.

## Overview

This project implements a regime detection system that:
- Identifies macroeconomic regimes (Equities, Fixed Income, Balanced, Cash) using rolling window estimation
- Compares two methodologies: K-Means vs HMM
- Provides tactical allocation recommendations based on detected regimes
- Includes a professional interactive dashboard built with Streamlit
- Ensures **no look-ahead bias** through proper data handling

## Quick Start

### 1. Download the project
- Click the green **"Code"** button above
- Select **"Download ZIP"**
- Unzip the folder

### 2. Open a terminal in the project folder
- **Mac**: Right-click on folder → "New Terminal at Folder"
- **Windows**: Open folder, type `cmd` in the address bar

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard
```bash
streamlit run dashboard_streamlit.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

That's it!

## Project Structure

```
macro-regime-detection/
├── dashboard_streamlit.py   # Interactive Streamlit dashboard
├── regime_detector.py       # Core detection algorithms
├── requirements.txt         # Python dependencies
├── .env                     # API configuration (included)
└── README.md                # This file
```

## Methodology

### Data Sources

| Variable | Source | Description |
|----------|--------|-------------|
| Unemployment Rate | FRED (UNRATE) | Labor market indicator |
| CPI | FRED (CPIAUCSL) | Inflation measure |
| 10Y Treasury | FRED (GS10) | Long-term rates |
| 2Y Treasury | FRED (GS2) | Short-term rates |
| VIX | FRED (VIXCLS) | Volatility index |
| BAA Yield | FRED (BAA) | Credit spread proxy |
| S&P 500 | Yahoo Finance | Equity benchmark |
| VBMFX | Yahoo Finance | Bond benchmark |

### Features

Derived features used in regime detection:
- `unemploy`: Unemployment rate level
- `infl_mom`: Inflation momentum (log CPI change)
- `ust10y_d`: 10Y yield monthly change
- `ust2y_d`: 2Y yield monthly change
- `2s10s_spread`: Yield curve slope (10Y - 2Y)
- `vix`: Volatility level
- `baa_yield`: Corporate yield level

### Regime Classification

| Regime | Condition | Allocation |
|--------|-----------|------------|
| **Equities** | Stocks outperform | 100% Equities |
| **Fixed Income** | Bonds outperform | 100% Bonds |
| **Balanced** | Both perform well | 60% Equities / 40% Bonds |
| **Cash** | Both underperform | 100% Cash |

### Anti-Look-Ahead Measures

1. **Feature lag**: All features shifted by 1 month
2. **Rolling estimation**: Model trained on [t-window, t-1] only
3. **Out-of-sample prediction**: Regime at t uses only prior data
4. **Confirmation rule**: N consecutive months required before regime change

## Dashboard Features

The interactive dashboard allows you to:

- **Adjust model parameters** in real-time:
  - Rolling window length (10-25 years)
  - Number of regimes (3-8)
  - Feature selection
  - Confirmation period

- **Compare methodologies**:
  - HMM vs K-Means regime detection
  - Stability analysis (regime switches count)
  - Agreement rate between methods

- **Analyze performance**:
  - Cumulative returns (log scale)
  - Risk metrics (CAGR, Sharpe, Max Drawdown)
  - Performance by regime

## Model Comparison

| Aspect | K-Means | HMM |
|--------|---------|-----|
| Temporal dynamics | None | Markov transitions |
| Interpretability | High | Moderate |
| Stability | Lower | Higher |
| Probability outputs | No | Yes |
| Computational cost | Low | Higher |

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependencies

## Limitations

1. **Detection lag**: Regime changes detected with delay
2. **Data revisions**: Macro data subject to revisions
3. **Past ≠ Future**: Historical performance not indicative of future results
4. **Educational purpose**: Not investment advice

## References

- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates"
