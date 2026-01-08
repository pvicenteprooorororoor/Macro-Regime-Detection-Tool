"""
================================================================================
MACRO REGIME DETECTION - PROFESSIONAL DASHBOARD
================================================================================
Interactive dashboard for macroeconomic regime detection
K-Means vs Hidden Markov Model comparison

Usage: streamlit run dashboard_streamlit.py
================================================================================
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# Data imports
try:
    from fredapi import Fred
    from dotenv import load_dotenv
    load_dotenv()
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Macro Regime Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple dark theme styling
st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #0f1419;
    }

    /* Headers */
    h1, h2, h3 {
        color: #e5e7eb;
    }

    /* Metric cards */
    .metric-container {
        background: #1a1f26;
        border: 1px solid #2a2f38;
        padding: 20px;
        margin: 8px 0;
    }

    .metric-label {
        color: #9199a1;
        font-size: 12px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #e5e7eb;
        font-size: 28px;
    }

    .metric-delta-positive {
        color: #6ee7b7;
        font-size: 14px;
    }

    .metric-delta-negative {
        color: #dc2626;
        font-size: 14px;
    }

    /* Regime banner */
    .regime-banner {
        background: #1a1f26;
        border-left: 3px solid;
        padding: 20px;
        margin: 16px 0;
    }

    /* Tables */
    .dataframe {
        font-size: 13px;
    }

    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f26;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #475569;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONSTANTS
# ==============================================================================

REGIME_COLORS = {
    "equities": "#3b82f6",
    "rates": "#10b981",
    "both": "#f59e0b",
    "none": "#94a3b8"
}

REGIME_LABELS = {
    "equities": "Equities",
    "rates": "Fixed Income",
    "both": "Balanced (60/40)",
    "none": "Cash / Risk-Off"
}

REGIME_ALLOCATIONS = {
    "equities": {"stocks": 100, "bonds": 0},
    "rates": {"stocks": 0, "bonds": 100},
    "both": {"stocks": 60, "bonds": 40},
    "none": {"stocks": 0, "bonds": 0}
}

# Available macro variables with descriptions
MACRO_VARIABLES = {
    "unemploy": {"name": "Unemployment Rate", "code": "UNRATE", "description": "Labor market health indicator"},
    "infl_mom": {"name": "Inflation Momentum", "code": "CPIAUCSL", "description": "Month-over-month CPI change"},
    "ust10y_d": {"name": "10Y Yield Change", "code": "GS10", "description": "Monthly change in 10Y Treasury"},
    "ust2y_d": {"name": "2Y Yield Change", "code": "GS2", "description": "Monthly change in 2Y Treasury"},
    "2s10s_spread": {"name": "Yield Curve (10Y-2Y)", "code": "GS10-GS2", "description": "Term spread, recession indicator"},
    "vix": {"name": "VIX", "code": "VIXCLS", "description": "Market volatility / fear gauge"},
    "baa_yield": {"name": "BAA Corporate Yield", "code": "BAA", "description": "Credit spread proxy"},
}

# ==============================================================================
# DATA LOADING
# ==============================================================================

@st.cache_data(ttl=3600)
def load_raw_data():
    """Load raw data from FRED and Yahoo Finance."""
    
    if not FRED_AVAILABLE:
        st.error("FRED API not available. Install: pip install fredapi python-dotenv")
        return None, None
    
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        st.error("FRED_API_KEY not found in .env file")
        return None, None
    
    fred = Fred(api_key=fred_key)
    
    start = "1986-01-01"
    
    # Load FRED series
    def get_series(code, name):
        try:
            s = fred.get_series(code, observation_start=start).to_frame(name)
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
            return pd.DataFrame()
    
    unemploy = get_series("UNRATE", "unemploy")
    cpi = get_series("CPIAUCSL", "cpi")
    ust10y = get_series("GS10", "ust10y")
    ust2y = get_series("GS2", "ust2y")
    vix = get_series("VIXCLS", "vix")
    baa_yield = get_series("BAA", "baa_yield")
    
    # Load Yahoo data
    def get_yahoo(ticker, name):
        try:
            df = yf.download(ticker, start=start, interval="1mo", progress=False, auto_adjust=False)
            if df.empty:
                return pd.Series(dtype=float)
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                s = close.squeeze()
            else:
                s = close
            s = s.dropna()
            s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
            s.name = name
            return s.sort_index()
        except Exception as e:
            st.warning(f"Could not load {ticker}: {e}")
            return pd.Series(dtype=float)
    
    spx = get_yahoo("^GSPC", "spx")
    bonds = get_yahoo("VBMFX", "bonds")
    
    # Combine macro data
    macro = pd.concat([
        unemploy.resample("ME").last(),
        cpi.resample("ME").last(),
        ust10y.resample("ME").mean(),
        ust2y.resample("ME").mean(),
        vix.resample("ME").mean(),
        baa_yield.resample("ME").mean(),
    ], axis=1).sort_index()
    
    macro = macro.interpolate(method="time", limit=2, limit_direction="forward").ffill()
    
    # Derived features
    if "cpi" in macro.columns:
        macro["infl_mom"] = np.log(macro["cpi"]).diff()
    if "ust10y" in macro.columns:
        macro["ust10y_d"] = macro["ust10y"].diff()
    if "ust2y" in macro.columns:
        macro["ust2y_d"] = macro["ust2y"].diff()
    if "ust10y" in macro.columns and "ust2y" in macro.columns:
        macro["2s10s_spread"] = macro["ust10y"] - macro["ust2y"]
    
    # Assets
    assets = pd.DataFrame({"spx": spx, "bonds": bonds})
    assets.index = pd.to_datetime(assets.index)
    
    return macro, assets

def prepare_features(macro, assets, selected_features):
    """Prepare features and targets without look-ahead bias."""
    
    available = [f for f in selected_features if f in macro.columns]
    
    # Shift features by 1 month (no look-ahead)
    X = macro[available].shift(1)
    X.index = X.index.to_period("M").to_timestamp("M")
    
    # Targets: next month returns
    spx_lr = np.log(assets["spx"]).diff()
    bonds_lr = np.log(assets["bonds"]).diff()
    
    spx_next = (np.exp(spx_lr.shift(-1)) - 1.0).rename("spx_next")
    bonds_next = (np.exp(bonds_lr.shift(-1)) - 1.0).rename("bonds_next")
    
    y = pd.concat([spx_next, bonds_next], axis=1)
    y.index = y.index.to_period("M").to_timestamp("M")
    
    # Align
    idx = X.index.intersection(y.dropna().index)
    X = X.loc[idx].dropna()
    y = y.loc[X.index]
    
    return X, y

# ==============================================================================
# MODEL FUNCTIONS
# ==============================================================================

def fit_hmm(X_train, n_states, random_state=42):
    """Fit Gaussian HMM."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train.values)
    
    best_model = None
    best_score = -np.inf
    
    for cov in ["full", "diag"]:
        for i in range(3):
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type=cov,
                    n_iter=200,
                    random_state=random_state + i,
                    tol=1e-3
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(Xs)
                score = model.score(Xs)
                if score > best_score:
                    best_model, best_score = model, score
            except:
                continue
        if best_model:
            break
    
    return best_model, scaler

def fit_kmeans(X_train, n_clusters, random_state=42):
    """Fit K-Means."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train.values)
    
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    model.fit(Xs)
    
    return model, scaler

def label_states(state_seq, y_train, n_states):
    """Label states based on asset performance."""
    labels = {}
    df = pd.DataFrame({"state": state_seq}, index=y_train.index)
    agg = df.join(y_train).groupby("state").median()
    
    threshold = 0.0015
    
    for s in range(n_states):
        if s not in agg.index:
            labels[s] = "none"
            continue
        row = agg.loc[s]
        spx_m = row.get("spx_next", 0)
        bond_m = row.get("bonds_next", 0)
        
        if (spx_m > 0) and (bond_m <= threshold):
            labels[s] = "equities"
        elif (bond_m > 0) and (spx_m <= threshold):
            labels[s] = "rates"
        elif (spx_m > 0) and (bond_m > threshold):
            labels[s] = "both"
        else:
            labels[s] = "none"
    
    return labels

@st.cache_data(show_spinner=False)
def run_regime_detection(_X, _y, window_years, n_states_hmm, n_clusters_kmeans, selected_features_tuple):
    """Run rolling regime detection for both models."""
    
    X = _X
    y = _y
    window = window_years * 12
    
    dates = X.index
    
    hmm_labels = []
    kmeans_labels = []
    hmm_raw_states = []
    kmeans_raw_states = []
    out_idx = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_iterations = len(dates) - window
    
    for i, t in enumerate(range(window, len(dates))):
        progress = (i + 1) / total_iterations
        progress_bar.progress(progress)
        status_text.text(f"Processing {dates[t].strftime('%Y-%m')}...")
        
        end = dates[t]
        tr_idx = dates[t - window:t]
        seq_idx = dates[t - window:t + 1]
        
        X_tr = X.loc[tr_idx].dropna()
        y_tr = y.loc[tr_idx].loc[X_tr.index]
        
        if len(X_tr) < max(n_states_hmm, n_clusters_kmeans) * 5:
            continue
        
        X_seq = X.loc[seq_idx]
        if X_seq.isna().any().any():
            continue
        
        # HMM
        try:
            hmm_model, hmm_scaler = fit_hmm(X_tr, n_states_hmm)
            if hmm_model:
                X_tr_s = hmm_scaler.transform(X_tr.values)
                states_tr = hmm_model.predict(X_tr_s)
                mapping = label_states(states_tr, y_tr, n_states_hmm)
                
                X_seq_s = hmm_scaler.transform(X_seq.values)
                probas = hmm_model.predict_proba(X_seq_s)
                s_t = int(np.argmax(probas[-1]))
                hmm_lab = mapping.get(s_t, "none")
                hmm_raw = s_t
            else:
                hmm_lab = "none"
                hmm_raw = -1
        except:
            hmm_lab = "none"
            hmm_raw = -1
        
        # K-Means
        try:
            km_model, km_scaler = fit_kmeans(X_tr, n_clusters_kmeans)
            X_tr_s = km_scaler.transform(X_tr.values)
            clusters_tr = km_model.predict(X_tr_s)
            mapping = label_states(clusters_tr, y_tr, n_clusters_kmeans)
            
            X_t_s = km_scaler.transform(X.loc[[end]].values)
            c_t = km_model.predict(X_t_s)[0]
            km_lab = mapping.get(c_t, "none")
            km_raw = c_t
        except:
            km_lab = "none"
            km_raw = -1
        
        hmm_labels.append(hmm_lab)
        kmeans_labels.append(km_lab)
        hmm_raw_states.append(hmm_raw)
        kmeans_raw_states.append(km_raw)
        out_idx.append(end)
    
    progress_bar.empty()
    status_text.empty()
    
    results = pd.DataFrame({
        "date": out_idx,
        "hmm_regime": hmm_labels,
        "kmeans_regime": kmeans_labels,
        "hmm_state": hmm_raw_states,
        "kmeans_cluster": kmeans_raw_states,
    })
    results = results.set_index("date")
    
    # Add returns
    results = results.join(y)
    
    return results

def apply_confirmation(labels, persist=2):
    """Apply confirmation rule to avoid whipsaw."""
    held = []
    current = None
    
    for i in range(len(labels)):
        if i + 1 >= persist:
            last_k = labels.iloc[i - persist + 1:i + 1]
            if (last_k.nunique() == 1) and (last_k.iloc[-1] != current):
                current = labels.iloc[i]
        held.append(current)
    
    return pd.Series(held, index=labels.index)

def calc_perf_stats(returns):
    """Calculate performance statistics."""
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return None
    
    n = len(returns)
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (12 / n) - 1 if n > 0 else 0
    vol = returns.std() * np.sqrt(12) if n > 1 else 0
    sharpe = cagr / vol if vol > 0 else 0
    
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() if len(dd) > 0 else 0
    
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "WinRate": (returns > 0).mean(),
        "N": n
    }

def count_regime_switches(labels):
    """Count number of regime switches."""
    labels = labels.dropna()
    if len(labels) < 2:
        return 0
    return (labels != labels.shift(1)).sum() - 1

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    # Header
    st.markdown("""
    <div style="padding: 20px 0 20px 0; border-bottom: 1px solid #2a2f38; margin-bottom: 30px;">
        <h1 style="margin: 0; font-size: 32px; color: #e5e7eb;">
            Macro Regime Detection
        </h1>
        <p style="margin: 12px 0 0 0; color: #d1d5db; font-size: 16px; line-height: 1.5;">
            This dashboard uses K-Means and HMM models to detect the current macroeconomic regime and assist with tactical allocation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Parameters
    with st.sidebar:
        st.markdown("""
        <h2 style="font-size: 18px; color: #e5e7eb; margin-bottom: 20px;">
            Model Parameters
        </h2>
        """, unsafe_allow_html=True)
        
        st.markdown("**Rolling Window**")
        window_years = st.slider("Window (years)", 10, 25, 20, 1,
                                  help="The rolling window defines the number of years of historical data used to train the model at each period. For example, with a 20-year window, the model is retrained each month on the last 20 years of data to predict the next month's regime.")
        
        st.markdown("---")
        
        st.markdown("**Number of Regimes**")
        n_regimes = st.slider("Regimes (both models)", 3, 8, 4, 1,
                              help="Same number used for HMM states and K-Means clusters for fair comparison")
        
        st.markdown("---")
        
        st.markdown("**Feature Selection**")
        
        selected_features = []
        for var_key, var_info in MACRO_VARIABLES.items():
            default = var_key in ["unemploy", "infl_mom", "ust10y_d", "2s10s_spread", "ust2y_d", "vix", "baa_yield"]
            if st.checkbox(var_info["name"], value=default, help=var_info["description"]):
                selected_features.append(var_key)
        
        st.markdown("---")

        st.markdown("**Confirmation Rule**")
        persistence = st.slider("Months to confirm", 1, 4, 1, 1,
                                help="The confirmation rule requires a new regime to be detected for N consecutive months before changing the allocation. This helps avoid excessive changes due to noise in the data (whipsaw).")
        
        st.markdown("---")
        
        run_button = st.button("Run Analysis", type="primary", use_container_width=True)
        
        # Date range info
        st.markdown("---")
        st.markdown("""
        <div style="padding: 16px; background: #1a1f26; border: 1px solid #2a2f38; margin-top: 20px;">
            <p style="color: #9199a1; font-size: 12px; margin: 0;">DATA PERIOD</p>
            <p style="color: #e5e7eb; font-size: 14px; margin: 4px 0 0 0;">1990 - Present</p>
            <p style="color: #9199a1; font-size: 11px; margin: 8px 0 0 0;">
                Sources: FRED, Yahoo Finance
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if not selected_features:
        st.warning("Please select at least one feature in the sidebar.")
        return
    
    # Load data
    with st.spinner("Loading market data..."):
        macro, assets = load_raw_data()
    
    if macro is None or assets is None:
        st.error("Failed to load data. Check your FRED API key.")
        return
    
    # Check for cached results or run analysis
    cache_key = f"{window_years}_{n_regimes}_{tuple(sorted(selected_features))}"
    
    if run_button or "results" not in st.session_state or st.session_state.get("cache_key") != cache_key:
        X, y = prepare_features(macro, assets, selected_features)
        
        if len(X) < window_years * 12 + 12:
            st.error("Not enough data for the selected window size.")
            return
        
        with st.spinner("Running regime detection..."):
            results = run_regime_detection(
                X, y, window_years, n_regimes, n_regimes, 
                tuple(sorted(selected_features))
            )
        
        # Apply confirmation rule
        results["hmm_held"] = apply_confirmation(results["hmm_regime"], persistence)
        results["kmeans_held"] = apply_confirmation(results["kmeans_regime"], persistence)
        
        # Calculate strategy returns
        for method in ["hmm", "kmeans"]:
            col = f"{method}_held"
            strat_returns = []
            for idx, row in results.iterrows():
                regime = row[col]
                if pd.isna(regime):
                    strat_returns.append(0)
                    continue
                alloc = REGIME_ALLOCATIONS.get(regime, REGIME_ALLOCATIONS["both"])
                ret = (alloc["stocks"]/100) * (row["spx_next"] or 0) + (alloc["bonds"]/100) * (row["bonds_next"] or 0)
                strat_returns.append(ret)
            results[f"{method}_return"] = strat_returns
        
        results["ret_60_40"] = 0.6 * results["spx_next"].fillna(0) + 0.4 * results["bonds_next"].fillna(0)
        
        # Cumulative returns
        results["cum_hmm"] = (1 + pd.Series(results["hmm_return"])).cumprod()
        results["cum_kmeans"] = (1 + pd.Series(results["kmeans_return"])).cumprod()
        results["cum_spx"] = (1 + results["spx_next"].fillna(0)).cumprod()
        results["cum_bonds"] = (1 + results["bonds_next"].fillna(0)).cumprod()
        results["cum_60_40"] = (1 + results["ret_60_40"]).cumprod()
        
        st.session_state["results"] = results
        st.session_state["cache_key"] = cache_key
    
    results = st.session_state["results"]
    
    # Current regime display
    current_date = results.index[-1]
    hmm_regime = results["hmm_held"].iloc[-1]
    kmeans_regime = results["kmeans_held"].iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        alloc = REGIME_ALLOCATIONS.get(hmm_regime, REGIME_ALLOCATIONS["both"])
        st.markdown(f"""
        <div class="regime-banner" style="border-color: {REGIME_COLORS.get(hmm_regime, '#94a3b8')};">
            <p class="metric-label" style="font-size: 14px;">HMM CURRENT REGIME</p>
            <p class="metric-value" style="color: {REGIME_COLORS.get(hmm_regime, '#94a3b8')}; font-size: 36px;">
                {REGIME_LABELS.get(hmm_regime, 'Unknown')}
            </p>
            <p style="color: #e5e7eb; font-size: 16px; margin-top: 16px; font-weight: 500;">
                Recommended Allocation: {alloc['stocks']}% Equities / {alloc['bonds']}% Fixed Income
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        alloc = REGIME_ALLOCATIONS.get(kmeans_regime, REGIME_ALLOCATIONS["both"])
        st.markdown(f"""
        <div class="regime-banner" style="border-color: {REGIME_COLORS.get(kmeans_regime, '#94a3b8')};">
            <p class="metric-label" style="font-size: 14px;">K-MEANS CURRENT REGIME</p>
            <p class="metric-value" style="color: {REGIME_COLORS.get(kmeans_regime, '#94a3b8')}; font-size: 36px;">
                {REGIME_LABELS.get(kmeans_regime, 'Unknown')}
            </p>
            <p style="color: #e5e7eb; font-size: 16px; margin-top: 16px; font-weight: 500;">
                Recommended Allocation: {alloc['stocks']}% Equities / {alloc['bonds']}% Fixed Income
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <p style="color: #94a3b8; font-size: 12px; text-align: right; margin-top: -10px;">
        As of {current_date.strftime('%B %d, %Y')}
    </p>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Model Comparison", "Regime Analysis", "Methodology"])
    
    # TAB 1: Performance
    with tab1:
        st.markdown("### Performance Summary")
        
        # Performance metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        strategies = [
            ("HMM Strategy", "hmm_return", "#60a5fa"),
            ("K-Means Strategy", "kmeans_return", "#5eead4"),
            ("S&P 500", "spx_next", "#3b82f6"),
            ("Fixed Income", "bonds_next", "#6ee7b7"),
            ("60/40 Benchmark", "ret_60_40", "#fcd34d"),
        ]
        
        for col, (name, ret_col, color) in zip([col1, col2, col3, col4, col5], strategies):
            with col:
                if ret_col in results.columns:
                    stats = calc_perf_stats(results[ret_col].tolist())
                    if stats:
                        cagr_color = "#6ee7b7" if stats["CAGR"] >= 0 else "#dc2626"
                        st.markdown(f"""
                        <div class="metric-container">
                            <p class="metric-label">{name}</p>
                            <p class="metric-value" style="color: {cagr_color};">{stats['CAGR']*100:.1f}%</p>
                            <p style="color: #94a3b8; font-size: 12px; margin-top: 8px;">
                                Sharpe: {stats['Sharpe']:.2f}<br>
                                Vol: {stats['Vol']*100:.1f}%<br>
                                Max DD: {stats['MaxDD']*100:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Cumulative performance chart
        st.markdown("### Cumulative Performance")
        
        fig = go.Figure()
        
        traces = [
            ("cum_hmm", "HMM Strategy", "#60a5fa", 3),
            ("cum_kmeans", "K-Means Strategy", "#5eead4", 3),
            ("cum_spx", "S&P 500", "#3b82f6", 1.5),
            ("cum_bonds", "Fixed Income", "#6ee7b7", 1.5),
            ("cum_60_40", "60/40 Benchmark", "#fcd34d", 1.5),
        ]
        
        for col_name, label, color, width in traces:
            if col_name in results.columns:
                fig.add_trace(go.Scatter(
                    x=results.index,
                    y=results[col_name],
                    name=label,
                    line=dict(color=color, width=width),
                    hovertemplate="%{y:.2f}<extra></extra>"
                ))
        
        fig.update_layout(
            height=450,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis_title="Growth of $1",
            yaxis_type="log",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=40, b=0)
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics table
        st.markdown("### Detailed Statistics")
        
        stats_data = []
        for name, ret_col, color in strategies:
            if ret_col in results.columns:
                stats = calc_perf_stats(results[ret_col].tolist())
                if stats:
                    stats_data.append({
                        "Strategy": name,
                        "CAGR": f"{stats['CAGR']*100:.2f}%",
                        "Volatility": f"{stats['Vol']*100:.2f}%",
                        "Sharpe Ratio": f"{stats['Sharpe']:.3f}",
                        "Max Drawdown": f"{stats['MaxDD']*100:.2f}%",
                        "Win Rate": f"{stats['WinRate']*100:.1f}%",
                        "Months": stats["N"]
                    })
        
        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    # TAB 2: Model Comparison
    with tab2:
        st.markdown("### Model Comparison: HMM vs K-Means")
        
        # Key comparison metrics
        col1, col2, col3 = st.columns(3)
        
        # Agreement rate
        valid_mask = results["hmm_held"].notna() & results["kmeans_held"].notna()
        agreement = (results.loc[valid_mask, "hmm_held"] == results.loc[valid_mask, "kmeans_held"]).mean() * 100
        
        # Regime switches
        hmm_switches = count_regime_switches(results["hmm_held"])
        kmeans_switches = count_regime_switches(results["kmeans_held"])
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-label">AGREEMENT RATE</p>
                <p class="metric-value">{agreement:.1f}%</p>
                <p style="color: #94a3b8; font-size: 12px; margin-top: 8px;">
                    Months with same regime detected
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-label">HMM REGIME SWITCHES</p>
                <p class="metric-value">{hmm_switches}</p>
                <p style="color: #94a3b8; font-size: 12px; margin-top: 8px;">
                    Total transitions over period
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-label">K-MEANS REGIME SWITCHES</p>
                <p class="metric-value">{kmeans_switches}</p>
                <p style="color: #94a3b8; font-size: 12px; margin-top: 8px;">
                    Total transitions over period
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Stability analysis
        st.markdown("### Regime Stability Analysis")
        st.markdown(f"""
        <div style="background: #1a1f26; border: 1px solid #2a2f38; padding: 20px; margin: 20px 0;">
            <p style="color: #e5e7eb; margin: 0;">
                <strong>Key Finding:</strong> K-Means shows <strong>{kmeans_switches}</strong> regime switches
                compared to HMM's <strong>{hmm_switches}</strong> switches.
                {'K-Means is more reactive and changes regimes more frequently.' if kmeans_switches > hmm_switches else 'HMM is more reactive and changes regimes more frequently.' if hmm_switches > kmeans_switches else 'Both models show similar stability.'}
            </p>
            <p style="color: #9199a1; margin: 12px 0 0 0; font-size: 13px;">
                HMM incorporates temporal dynamics through its transition matrix, which naturally smooths regime changes.
                K-Means treats each observation independently, leading to potentially more volatile classifications.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Side-by-side regime timeline
        st.markdown("### Regime Timeline Comparison")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("HMM Regimes", "K-Means Regimes"),
            row_heights=[0.5, 0.5]
        )
        
        regime_order = {"none": 0, "rates": 1, "both": 2, "equities": 3}
        
        # Track which regimes have been added to legend
        legend_added = set()
        
        # HMM regimes (row 1)
        for regime, code in regime_order.items():
            mask = results["hmm_held"] == regime
            if mask.any():
                show_legend = regime not in legend_added
                fig.add_trace(
                    go.Bar(
                        x=results.index[mask],
                        y=[1] * mask.sum(),
                        name=f"{REGIME_LABELS.get(regime, regime)}",
                        marker_color=REGIME_COLORS.get(regime, "#888"),
                        showlegend=show_legend,
                        legendgroup=regime
                    ),
                    row=1, col=1
                )
                legend_added.add(regime)
        
        # K-Means regimes (row 2)
        for regime, code in regime_order.items():
            mask = results["kmeans_held"] == regime
            if mask.any():
                show_legend = regime not in legend_added
                fig.add_trace(
                    go.Bar(
                        x=results.index[mask],
                        y=[1] * mask.sum(),
                        name=f"{REGIME_LABELS.get(regime, regime)}",
                        marker_color=REGIME_COLORS.get(regime, "#888"),
                        showlegend=show_legend,
                        legendgroup=regime
                    ),
                    row=2, col=1
                )
                legend_added.add(regime)
        
        fig.update_layout(
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="stack",
            bargap=0,
            legend=dict(orientation="h", yanchor="bottom", y=1.08),
            margin=dict(l=0, r=0, t=60, b=0)
        )
        
        fig.update_yaxes(showticklabels=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw state/cluster visualization
        st.markdown("### Raw State/Cluster Evolution")
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=results.index,
            y=results["hmm_state"],
            name="HMM State",
            mode="lines",
            line=dict(color="#60a5fa", width=2),
        ))

        fig2.add_trace(go.Scatter(
            x=results.index,
            y=results["kmeans_cluster"],
            name="K-Means Cluster",
            mode="lines",
            line=dict(color="#5eead4", width=2),
        ))
        
        fig2.update_layout(
            height=300,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="State / Cluster ID",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Comparison table
        st.markdown("### Method Characteristics")
        
        comparison_df = pd.DataFrame({
            "Characteristic": [
                "Temporal Dynamics",
                "Interpretability", 
                "Computational Cost",
                "Stability",
                "Probability Outputs",
                "Transition Modeling"
            ],
            "HMM": [
                "Captures time dependencies",
                "Moderate - requires understanding of Markov chains",
                "Higher - iterative EM algorithm",
                "More stable - smoothed by transition matrix",
                "Yes - posterior probabilities available",
                "Explicit transition matrix A"
            ],
            "K-Means": [
                "None - treats observations independently",
                "High - simple cluster assignments",
                "Lower - fast convergence",
                "Less stable - can oscillate",
                "No - hard cluster assignments only",
                "None"
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # TAB 3: Regime Analysis
    with tab3:
        st.markdown("### Regime Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**HMM Regime Distribution**")
            hmm_dist = results["hmm_held"].value_counts(normalize=True) * 100
            
            for regime, pct in hmm_dist.items():
                if regime and not pd.isna(regime):
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin: 8px 0;">
                        <div style="width: 12px; height: 12px; background: {REGIME_COLORS.get(regime, '#94a3b8')};
                                    border-radius: 2px; margin-right: 12px;"></div>
                        <span style="color: #e5e7eb; flex: 1;">{REGIME_LABELS.get(regime, regime)}</span>
                        <span style="color: #94a3b8;">{pct:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            st.markdown("**K-Means Regime Distribution**")
            km_dist = results["kmeans_held"].value_counts(normalize=True) * 100

            for regime, pct in km_dist.items():
                if regime and not pd.isna(regime):
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin: 8px 0;">
                        <div style="width: 12px; height: 12px; background: {REGIME_COLORS.get(regime, '#94a3b8')};
                                    border-radius: 2px; margin-right: 12px;"></div>
                        <span style="color: #e5e7eb; flex: 1;">{REGIME_LABELS.get(regime, regime)}</span>
                        <span style="color: #94a3b8;">{pct:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Asset performance by regime
        st.markdown("### Asset Performance by Regime (HMM)")
        
        perf_data = []
        for regime in ["equities", "rates", "both", "none"]:
            mask = results["hmm_held"] == regime
            if mask.sum() > 0:
                spx_ret = results.loc[mask, "spx_next"].mean() * 12 * 100
                bond_ret = results.loc[mask, "bonds_next"].mean() * 12 * 100
                perf_data.append({
                    "Regime": REGIME_LABELS.get(regime, regime),
                    "Equities (Ann.)": spx_ret,
                    "Fixed Income (Ann.)": bond_ret,
                    "Months": mask.sum()
                })
        
        if perf_data:
            fig = go.Figure()
            
            perf_df = pd.DataFrame(perf_data)
            
            fig.add_trace(go.Bar(
                name="Equities",
                x=perf_df["Regime"],
                y=perf_df["Equities (Ann.)"],
                marker_color="#3b82f6"
            ))

            fig.add_trace(go.Bar(
                name="Fixed Income",
                x=perf_df["Regime"],
                y=perf_df["Fixed Income (Ann.)"],
                marker_color="#6ee7b7"
            ))
            
            fig.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                barmode="group",
                yaxis_title="Annualized Return (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=0)
            )

            fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Methodology
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Hidden Markov Model
            
            **Mathematical Framework**
            
            A Hidden Markov Model assumes that the system transitions between 
            hidden states according to a Markov chain, with observations generated 
            from state-dependent emission distributions.
            
            **Components:**
            - π: Initial state distribution
            - A: Transition matrix P(Sₜ₊₁|Sₜ)
            - B: Emission distributions N(μᵢ, Σᵢ)
            
            **Estimation:** Baum-Welch (EM algorithm)
            
            **Decoding:** Viterbi algorithm for optimal state sequence
            
            **Advantages:**
            - Captures regime persistence
            - Provides transition probabilities
            - Smooth regime changes
            
            **Limitations:**
            - Computationally intensive
            - Assumes stationary parameters
            - Sensitive to initialization
            """)
        
        with col2:
            st.markdown("""
            ### K-Means Clustering
            
            **Mathematical Framework**
            
            K-Means partitions observations into K clusters by minimizing 
            within-cluster variance (inertia):
            
            J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
            
            **Algorithm (Lloyd's):**
            1. Initialize K centroids
            2. Assign points to nearest centroid
            3. Update centroids as cluster means
            4. Repeat until convergence
            
            **Advantages:**
            - Simple and interpretable
            - Fast computation
            - Centroids represent regime profiles
            
            **Limitations:**
            - No temporal structure
            - Assumes spherical clusters
            - Hard assignments only
            - Can be unstable
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### Variables Used
        """)
        
        var_df = pd.DataFrame([
            {"Variable": v["name"], "Source": v["code"], "Description": v["description"]}
            for k, v in MACRO_VARIABLES.items()
            if k in selected_features
        ])
        
        st.dataframe(var_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ### Anti-Look-Ahead Measures
        
        1. **Feature Lag:** All features are shifted by 1 month (Xₜ uses information up to t-1)
        2. **Rolling Estimation:** Model trained on [t-window, t-1] only
        3. **Out-of-Sample Prediction:** Regime at t predicted using model trained on prior data
        4. **Confirmation Rule:** Regime change requires N consecutive months of agreement
        """)

if __name__ == "__main__":
    main()
