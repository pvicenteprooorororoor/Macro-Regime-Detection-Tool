"""
================================================================================
MACRO REGIME DETECTION TOOL - K-MEANS vs HMM
================================================================================
Outil complet de détection de régimes macroéconomiques
Comparaison K-Means Clustering vs Hidden Markov Models (HMM)
Sans biais de look-ahead (rolling window estimation)

Basé sur l'approche initiale avec ajout de K-Means pour comparaison.

Requirements:
    pip install pandas numpy yfinance fredapi python-dotenv scikit-learn hmmlearn matplotlib seaborn plotly

Usage:
    python regime_detector.py
    
Author: Senior Quant Macro
================================================================================
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Data fetching
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("⚠️ fredapi non installé. pip install fredapi")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("⚠️ yfinance non installé. pip install yfinance")

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️ hmmlearn non installé. pip install hmmlearn")

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Configuration centralisée du modèle."""
    
    # Dates
    START_DATE: str = "1986-01-01"
    
    # Fenêtre rolling (années et mois)
    WINDOW_YEARS: int = 20
    WINDOW_MONTHS: int = field(init=False)
    
    # Nombre d'états/clusters
    N_STATES_HMM: int = 5
    N_CLUSTERS_KMEANS: int = 4
    
    # Règle de confirmation (mois)
    PERSISTENCE: int = 2
    
    # HMM parameters
    HMM_COV_TYPE: str = "full"
    HMM_N_ITER: int = 300
    HMM_N_INIT: int = 5
    
    # K-Means parameters
    KMEANS_N_INIT: int = 10
    KMEANS_MAX_ITER: int = 300
    
    # Random seed
    RANDOM_STATE: int = 42
    
    # Features à utiliser
    FEATURES: List[str] = field(default_factory=lambda: [
        "unemploy", "infl_mom", "ust10y_d", "2s10s_spread", 
        "ust2y_d", "vix", "baa_yield"
    ])
    
    def __post_init__(self):
        self.WINDOW_MONTHS = self.WINDOW_YEARS * 12


# ==============================================================================
# DATA LOADER
# ==============================================================================

class MacroDataLoader:
    """
    Chargement des données macroéconomiques et d'actifs.
    
    Variables macroéconomiques sélectionnées:
    ==========================================
    
    1. UNRATE (Unemployment Rate) - Taux de chômage
       → Indicateur retardé du cycle économique
       → Reflète la santé du marché du travail
       → Impact sur la consommation et la politique monétaire
    
    2. CPIAUCSL (CPI All Urban Consumers) - Inflation
       → Mesure clé pour la Fed
       → Impact sur les taux réels et les valorisations
       → Dérivée: infl_mom = momentum d'inflation
    
    3. GS10 (10-Year Treasury Constant Maturity)
       → Benchmark des taux longs
       → Reflète les anticipations de croissance et d'inflation
       → Dérivée: ust10y_d = variation mensuelle
    
    4. GS2 (2-Year Treasury Constant Maturity)
       → Reflète les anticipations de politique monétaire
       → Dérivée: ust2y_d = variation mensuelle
       → Combiné: 2s10s_spread = pente de la courbe (GS10 - GS2)
    
    5. VIXCLS (CBOE Volatility Index)
       → Mesure de l'aversion au risque
       → "Fear gauge" du marché
       → Corrélé négativement aux rendements actions
    
    6. BAA (Moody's BAA Corporate Bond Yield)
       → Prime de risque corporate
       → Indicateur de stress sur le crédit
       → Conditions de financement des entreprises
    
    Actifs pour le backtest:
    ========================
    
    1. ^GSPC (S&P 500)
       → Proxy actions US le plus liquide
       → Représentatif du marché large
    
    2. VBMFX (Vanguard Total Bond Market Index)
       → Proxy obligations aggregate US
       → Duration moyenne ~6 ans
       → Diversification vs actions
    """
    
    # Séries FRED
    FRED_SERIES = {
        "unemploy": "UNRATE",      # Unemployment Rate
        "cpi": "CPIAUCSL",         # CPI All Urban Consumers
        "ust10y": "GS10",          # 10Y Treasury Yield
        "ust2y": "GS2",            # 2Y Treasury Yield
        "vix": "VIXCLS",           # VIX Index
        "baa_yield": "BAA",        # BAA Corporate Yield
    }
    
    # Tickers Yahoo
    YAHOO_TICKERS = {
        "spx": "^GSPC",            # S&P 500
        "bonds": "VBMFX",          # Vanguard Total Bond
    }
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialise le loader.
        
        Args:
            fred_api_key: Clé API FRED. Si None, cherche dans .env
        """
        load_dotenv()
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        
        if FRED_AVAILABLE and self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            self.fred = None
            print("⚠️ FRED API non configurée")
    
    def get_fred_series(self, code: str, name: str, start: str) -> pd.DataFrame:
        """Récupère une série FRED."""
        if self.fred is None:
            raise ValueError("FRED API non configurée")
        s = self.fred.get_series(code, observation_start=start).to_frame(name)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    
    def get_yahoo_monthly(self, ticker: str, start: str) -> pd.Series:
        """Récupère les cours mensuels de Yahoo Finance."""
        if not YF_AVAILABLE:
            raise ImportError("yfinance requis")
        
        df = yf.download(ticker, start=start, interval="1mo", progress=False, auto_adjust=False)
        if df.empty:
            raise RuntimeError(f"Données introuvables pour {ticker}")
        
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            s = close[ticker] if ticker in close.columns else close.squeeze()
        else:
            s = close
        
        s = s.dropna()
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        s.name = ticker
        return s.sort_index()
    
    def load_all_data(self, start: str = "1986-01-01") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Charge toutes les données macro et actifs.
        
        Returns:
            Tuple[macro_df, assets_df]
        """
        print("📊 Chargement des données...")
        
        # === MACRO DATA (FRED) ===
        macro_series = {}
        for name, code in self.FRED_SERIES.items():
            try:
                df = self.get_fred_series(code, name, start)
                macro_series[name] = df
                print(f"  ✓ {name} ({code}): {len(df)} points")
            except Exception as e:
                print(f"  ✗ {name} ({code}): {e}")
        
        # === ASSET DATA (Yahoo) ===
        asset_series = {}
        for name, ticker in self.YAHOO_TICKERS.items():
            try:
                s = self.get_yahoo_monthly(ticker, start)
                asset_series[name] = s
                print(f"  ✓ {name} ({ticker}): {len(s)} points")
            except Exception as e:
                print(f"  ✗ {name} ({ticker}): {e}")
        
        # === ASSEMBLAGE ===
        # Macro: resampling mensuel
        def monthly_mean(df):
            return df.resample("ME").mean()
        
        def monthly_last(df):
            return df.resample("ME").last()
        
        macro = pd.concat([
            monthly_last(macro_series.get("unemploy", pd.DataFrame())),
            monthly_last(macro_series.get("cpi", pd.DataFrame())),
            monthly_mean(macro_series.get("ust10y", pd.DataFrame())),
            monthly_mean(macro_series.get("ust2y", pd.DataFrame())),
            monthly_mean(macro_series.get("vix", pd.DataFrame())),
            monthly_mean(macro_series.get("baa_yield", pd.DataFrame())),
        ], axis=1).sort_index()
        
        # Imputation forward only (pas de look-ahead)
        macro = macro.interpolate(method="time", limit=2, limit_direction="forward").ffill()
        
        # Align sur les séries disponibles
        core_cols = [c for c in ["unemploy", "cpi", "ust10y", "ust2y", "vix", "baa_yield"] 
                     if c in macro.columns]
        start_all = max(macro[c].first_valid_index() for c in core_cols)
        macro = macro[macro.index >= start_all].copy()
        
        # Assets
        assets = pd.DataFrame({
            "spx_price": asset_series.get("spx"),
            "bond_price": asset_series.get("bonds"),
        })
        assets.index = pd.to_datetime(assets.index)
        
        print(f"\n✓ Macro: {macro.shape[0]} mois, {macro.shape[1]} variables")
        print(f"✓ Assets: {assets.shape[0]} mois")
        
        return macro, assets
    
    def prepare_features(self, macro: pd.DataFrame, assets: pd.DataFrame, 
                         config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prépare les features et targets sans look-ahead bias.
        
        Anti-look-ahead measures:
        1. Features shiftées de 1 mois (X_t utilise info jusqu'à t-1)
        2. Targets = rendements du mois SUIVANT
        
        Returns:
            Tuple[X, y] - Features et targets alignés
        """
        print("\n🔧 Préparation des features...")
        
        # === DERIVED FEATURES ===
        macro = macro.copy()
        
        # Momentum d'inflation (log return mensuel du CPI)
        if "cpi" in macro.columns:
            macro["infl_mom"] = np.log(macro["cpi"]).diff()
        
        # Variations des taux
        for col in ["ust10y", "ust2y", "vix"]:
            if col in macro.columns:
                macro[col + "_d"] = macro[col].diff()
        
        # Spread de courbe (pente)
        if "ust10y" in macro.columns and "ust2y" in macro.columns:
            macro["2s10s_spread"] = macro["ust10y"] - macro["ust2y"]
        
        # === FEATURES MATRIX ===
        available_features = [f for f in config.FEATURES if f in macro.columns]
        print(f"  Features disponibles: {available_features}")
        
        # Shift de 1 mois pour éviter look-ahead
        X = macro[available_features].shift(1)
        X.index = X.index.to_period("M").to_timestamp("M")
        
        # === TARGETS (rendements du mois suivant) ===
        if "spx_price" in assets.columns:
            spx_lr = np.log(assets["spx_price"]).diff()
            spx_next = (np.exp(spx_lr.shift(-1)) - 1.0).rename("spx_next")
        else:
            spx_next = pd.Series(dtype=float, name="spx_next")
        
        if "bond_price" in assets.columns:
            bond_lr = np.log(assets["bond_price"]).diff()
            bond_next = (np.exp(bond_lr.shift(-1)) - 1.0).rename("vbmfx_next")
        else:
            bond_next = pd.Series(dtype=float, name="vbmfx_next")
        
        y = pd.concat([spx_next, bond_next], axis=1)
        y.index = y.index.to_period("M").to_timestamp("M")
        
        # === ALIGNEMENT ===
        idx = X.index.intersection(y.dropna().index)
        X = X.loc[idx].dropna()
        y = y.loc[X.index]
        
        print(f"  ✓ X: {X.shape[0]} mois x {X.shape[1]} features")
        print(f"  ✓ y: {y.shape[0]} mois x {y.shape[1]} targets")
        print(f"  ✓ Période: {X.index.min().date()} → {X.index.max().date()}")
        
        return X, y


# ==============================================================================
# HMM CONFIG
# ==============================================================================

@dataclass
class HMMConfig:
    """Configuration du modèle HMM."""
    n_states: int = 5
    cov_type: str = "full"
    n_iter: int = 300
    n_init: int = 5
    random_state: int = 42
    min_covar: float = 1e-6


# ==============================================================================
# REGIME DETECTOR - HMM
# ==============================================================================

class HMMRegimeDetector:
    """
    Détection de régimes par Hidden Markov Model.
    
    HIDDEN MARKOV MODEL - Description Mathématique
    ===============================================
    
    Un HMM est un modèle génératif probabiliste où:
    - Les états (régimes) sont CACHÉS (latents)
    - Seules les observations (variables macro) sont visibles
    - Les états évoluent selon une chaîne de Markov
    
    Composantes du modèle:
    
    1. Distribution initiale π:
       πᵢ = P(S₁ = sᵢ)
       Probabilité d'être dans l'état i au temps 1
    
    2. Matrice de transition A (K×K):
       aᵢⱼ = P(Sₜ₊₁ = sⱼ | Sₜ = sᵢ)
       Probabilité de passer de l'état i à j
       → Capture la PERSISTANCE des régimes
    
    3. Distributions d'émission B (Gaussiennes multivariées):
       bᵢ(x) = N(x | μᵢ, Σᵢ)
       Distribution des observations dans l'état i
       → μᵢ = profil moyen du régime i
       → Σᵢ = covariance (corrélations entre variables)
    
    Estimation (Baum-Welch / EM):
    
    1. E-step: Forward-Backward algorithm
       - Forward: αₜ(i) = P(X₁:ₜ, Sₜ = sᵢ | λ)
       - Backward: βₜ(i) = P(Xₜ₊₁:T | Sₜ = sᵢ, λ)
       - Posterior: γₜ(i) = P(Sₜ = sᵢ | X₁:T, λ)
    
    2. M-step: Mise à jour des paramètres
       π̂ᵢ = γ₁(i)
       âᵢⱼ = Σₜ ξₜ(i,j) / Σₜ γₜ(i)
       μ̂ᵢ, Σ̂ᵢ via weighted MLE
    
    Décodage (Viterbi):
    S* = argmax P(S₁:T | X₁:T, λ)
    → Séquence d'états la plus probable
    
    Choix du nombre d'états (K):
    - BIC = -2·log(L) + K·log(T)
    - AIC = -2·log(L) + 2K
    - K=5 choisi pour capturer:
      * Expansion forte
      * Expansion modérée
      * Ralentissement
      * Récession
      * Crise/stress extrême
    
    Avantages pour l'allocation tactique:
    + Capture la PERSISTANCE des régimes (vs K-Means)
    + Fournit des PROBABILITÉS (incertitude quantifiée)
    + Transitions LISSES entre régimes
    + Prédiction forward possible via A
    
    Limites:
    - Complexité computationnelle
    - Sensibilité à l'initialisation
    - Assume stationnarité (paramètres fixes)
    - Peut être instable avec peu de données
    """
    
    def __init__(self, config: HMMConfig):
        self.config = config
        
    def fit_hmm(self, X_train: pd.DataFrame) -> Tuple[GaussianHMM, StandardScaler]:
        """
        Fit HMM avec multi-initialisation pour robustesse.
        
        Returns:
            Tuple[model, scaler]
        """
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train.values)
        
        best_model = None
        best_score = -np.inf
        
        # Essayer plusieurs types de covariance (fallback si échec)
        cov_order = [self.config.cov_type]
        if self.config.cov_type != "diag":
            cov_order.append("diag")
        
        for cov in cov_order:
            for i in range(max(1, self.config.n_init)):
                seed = (self.config.random_state or 0) + i
                try:
                    model = GaussianHMM(
                        n_components=self.config.n_states,
                        covariance_type=cov,
                        n_iter=self.config.n_iter,
                        random_state=seed,
                        tol=1e-3,
                        min_covar=self.config.min_covar,
                        verbose=False,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(Xs)
                    
                    score = model.score(Xs)
                    if score > best_score:
                        best_model, best_score = model, score
                except Exception:
                    continue
            
            if best_model is not None:
                break
        
        if best_model is None:
            raise RuntimeError("HMM fit failed")
        
        return best_model, scaler
    
    def label_states(self, state_seq: np.ndarray, y_train: pd.DataFrame) -> Dict[int, str]:
        """
        Labelling des états HMM basé sur les rendements médians.
        
        Logique de labelling (basée sur les rendements in-sample):
        - equities: rendement actions > 0 ET rendement bonds faible
        - rates: rendement bonds > 0 ET rendement actions faible
        - both: les deux > seuil (environnement favorable)
        - none: les deux faibles/négatifs (cash)
        
        Seuil de 15bp mensuel (~1.8% annualisé) pour distinguer les régimes.
        """
        labels = {}
        df = pd.DataFrame({"state": state_seq}, index=y_train.index)
        agg = df.join(y_train).groupby("state").median()
        
        threshold = 0.0015  # 15bp mensuel
        
        for s, row in agg.iterrows():
            spx_m = row.get("spx_next", 0)
            bond_m = row.get("vbmfx_next", 0)
            
            if (spx_m > 0) and (bond_m <= threshold):
                labels[s] = "equities"
            elif (bond_m > 0) and (spx_m <= threshold):
                labels[s] = "rates"
            elif (spx_m > 0) and (bond_m > threshold):
                labels[s] = "both"
            else:
                labels[s] = "none"
        
        return labels
    
    def rolling_predict(self, X: pd.DataFrame, y: pd.DataFrame, 
                        window: int) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Prédiction rolling sans look-ahead.
        
        À chaque date t:
        1. Entraînement sur [t-window, t-1]
        2. Labelling via rendements sur la fenêtre d'entraînement
        3. Prédiction à t via posterior sur [t-window, t]
        
        Returns:
            Tuple[labels_series, pmax_df, probas_df]
        """
        dates = X.index
        out_labels = []
        out_idx = []
        
        pmax_mean_list = []
        pmax_min_list = []
        pmax_t_list = []
        
        # Pour stocker les probabilités de chaque état
        state_probas = {s: [] for s in range(self.config.n_states)}
        
        print(f"\n🔄 Rolling HMM prediction (window={window} mois)...")
        
        for t in range(window, len(dates)):
            if t % 50 == 0:
                print(f"  Progress: {t}/{len(dates)} ({100*t/len(dates):.1f}%)")
            
            end = dates[t]
            tr_idx = dates[t - window:t]      # [t-window, t-1] pour train
            seq_idx = dates[t - window:t + 1]  # [t-window, t] pour posterior
            
            X_tr = X.loc[tr_idx].dropna()
            y_tr = y.loc[tr_idx].loc[X_tr.index]
            
            if len(X_tr) < self.config.n_states * 5:
                continue
            
            X_seq = X.loc[seq_idx]
            if X_seq.isna().any().any():
                continue
            
            try:
                model, scaler = self.fit_hmm(X_tr)
                
                # Labelling sur train
                X_tr_s = scaler.transform(X_tr.values)
                states_tr = model.predict(X_tr_s)
                mapping = self.label_states(states_tr, y_tr)
                
                # Probas sur train pour diagnostics
                probas_tr = model.predict_proba(X_tr_s)
                pmax_tr = probas_tr.max(axis=1)
                pmax_train_mean = float(np.mean(pmax_tr))
                pmax_train_min = float(np.min(pmax_tr))
                
                # Posterior sur séquence complète
                X_seq_s = scaler.transform(X_seq.values)
                probas_seq = model.predict_proba(X_seq_s)
                
                # Proba au point t
                p_t = probas_seq[-1]
                pmax_t = float(p_t.max())
                s_t = int(np.argmax(p_t))
                lab_t = mapping.get(s_t, "none")
                
                # Stocker les probas par état
                for s in range(self.config.n_states):
                    state_probas[s].append(p_t[s] if s < len(p_t) else 0)
                
            except Exception:
                continue
            
            out_labels.append(lab_t)
            out_idx.append(end)
            pmax_mean_list.append(pmax_train_mean)
            pmax_min_list.append(pmax_train_min)
            pmax_t_list.append(pmax_t)
        
        labels_series = pd.Series(out_labels, index=pd.Index(out_idx, name="date"), name="label_pred")
        
        pmax_df = pd.DataFrame({
            "pmax_train_mean": pmax_mean_list,
            "pmax_train_min": pmax_min_list,
            "pmax_t": pmax_t_list,
        }, index=labels_series.index)
        
        # DataFrame des probabilités par état
        probas_df = pd.DataFrame(
            {f"state_{s}_prob": state_probas[s] for s in range(self.config.n_states)},
            index=labels_series.index
        )
        
        print(f"  ✓ {len(labels_series)} prédictions générées")
        
        return labels_series, pmax_df, probas_df


# ==============================================================================
# REGIME DETECTOR - K-MEANS
# ==============================================================================

class KMeansRegimeDetector:
    """
    Détection de régimes par K-Means Clustering.
    
    K-MEANS CLUSTERING - Description Mathématique
    =============================================
    
    K-Means partitionne n observations en K clusters en minimisant
    la variance intra-cluster (inertie).
    
    Fonction objectif:
    
    J = Σᵢ₌₁ᴷ Σₓ∈Cᵢ ||x - μᵢ||²
    
    où:
    - Cᵢ = cluster i
    - μᵢ = centroïde du cluster i (moyenne)
    - ||·|| = distance euclidienne
    
    Algorithme de Lloyd:
    
    1. Initialisation: K centroïdes aléatoires (ou K-means++)
    2. Assignment: Chaque point → centroïde le plus proche
       cᵢ = argmin_j ||xᵢ - μⱼ||²
    3. Update: Recalcul des centroïdes
       μⱼ = (1/|Cⱼ|) Σₓ∈Cⱼ x
    4. Répéter 2-3 jusqu'à convergence
    
    Complexité: O(n·K·I·d) où I=itérations, d=dimensions
    
    Choix de K (nombre de clusters):
    
    1. Méthode du coude (Elbow):
       - Plot inertie vs K
       - Chercher le "coude" où la réduction marginale diminue
    
    2. Silhouette Score:
       s(i) = (b(i) - a(i)) / max(a(i), b(i))
       - a(i) = distance moyenne intra-cluster
       - b(i) = distance moyenne au cluster le plus proche
       - s ∈ [-1, 1], plus élevé = meilleur
    
    3. Pour notre cas: K=4 pour capturer:
       * Expansion (GDP↑, VIX↓)
       * Récession (GDP↓, VIX↑)
       * Slowdown (GDP~0, Inflation↑)
       * Recovery (GDP↑ depuis bas, VIX↓)
    
    Avantages:
    + Simple et interprétable
    + Centroïdes = profil type de chaque régime
    + Rapide (O(n·K))
    + Stable avec K-means++ init
    
    Limites:
    - Pas de structure TEMPORELLE (chaque point indépendant)
    - Assume clusters SPHÉRIQUES (même variance)
    - Sensible aux outliers
    - Pas de probabilités natives
    
    Comparaison avec HMM:
    - K-Means: snapshot statique des conditions
    - HMM: séquence temporelle avec transitions
    """
    
    def __init__(self, n_clusters: int = 4, n_init: int = 10, 
                 max_iter: int = 300, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit_kmeans(self, X_train: pd.DataFrame) -> Tuple[KMeans, StandardScaler]:
        """Fit K-Means sur les données d'entraînement."""
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train.values)
        
        model = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        model.fit(Xs)
        
        return model, scaler
    
    def label_clusters(self, cluster_seq: np.ndarray, y_train: pd.DataFrame) -> Dict[int, str]:
        """
        Labelling des clusters basé sur les rendements médians.
        Même logique que pour HMM.
        """
        labels = {}
        df = pd.DataFrame({"cluster": cluster_seq}, index=y_train.index)
        agg = df.join(y_train).groupby("cluster").median()
        
        threshold = 0.0015
        
        for c, row in agg.iterrows():
            spx_m = row.get("spx_next", 0)
            bond_m = row.get("vbmfx_next", 0)
            
            if (spx_m > 0) and (bond_m <= threshold):
                labels[c] = "equities"
            elif (bond_m > 0) and (spx_m <= threshold):
                labels[c] = "rates"
            elif (spx_m > 0) and (bond_m > threshold):
                labels[c] = "both"
            else:
                labels[c] = "none"
        
        return labels
    
    def rolling_predict(self, X: pd.DataFrame, y: pd.DataFrame,
                        window: int) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Prédiction rolling K-Means sans look-ahead.
        
        Returns:
            Tuple[labels_series, distances_df]
        """
        dates = X.index
        out_labels = []
        out_idx = []
        
        distance_to_centroid = []
        silhouette_scores = []
        
        print(f"\n🔄 Rolling K-Means prediction (window={window} mois)...")
        
        for t in range(window, len(dates)):
            if t % 50 == 0:
                print(f"  Progress: {t}/{len(dates)} ({100*t/len(dates):.1f}%)")
            
            end = dates[t]
            tr_idx = dates[t - window:t]
            
            X_tr = X.loc[tr_idx].dropna()
            y_tr = y.loc[tr_idx].loc[X_tr.index]
            
            if len(X_tr) < self.n_clusters * 5:
                continue
            
            X_t = X.loc[[end]]
            if X_t.isna().any().any():
                continue
            
            try:
                model, scaler = self.fit_kmeans(X_tr)
                
                # Labelling sur train
                X_tr_s = scaler.transform(X_tr.values)
                clusters_tr = model.predict(X_tr_s)
                mapping = self.label_clusters(clusters_tr, y_tr)
                
                # Prédiction au point t
                X_t_s = scaler.transform(X_t.values)
                cluster_t = model.predict(X_t_s)[0]
                lab_t = mapping.get(cluster_t, "none")
                
                # Distance au centroïde (mesure de confiance)
                distances = np.linalg.norm(model.cluster_centers_ - X_t_s, axis=1)
                dist_t = distances[cluster_t]
                
                # Silhouette score sur train (qualité du clustering)
                if len(np.unique(clusters_tr)) > 1:
                    sil = silhouette_score(X_tr_s, clusters_tr)
                else:
                    sil = 0
                
            except Exception:
                continue
            
            out_labels.append(lab_t)
            out_idx.append(end)
            distance_to_centroid.append(dist_t)
            silhouette_scores.append(sil)
        
        labels_series = pd.Series(out_labels, index=pd.Index(out_idx, name="date"), name="label_pred")
        
        metrics_df = pd.DataFrame({
            "distance_to_centroid": distance_to_centroid,
            "silhouette_score": silhouette_scores,
        }, index=labels_series.index)
        
        print(f"  ✓ {len(labels_series)} prédictions générées")
        
        return labels_series, metrics_df


# ==============================================================================
# BACKTEST ENGINE
# ==============================================================================

class BacktestEngine:
    """
    Moteur de backtest pour évaluer les stratégies de régimes.
    """
    
    @staticmethod
    def apply_confirmation(labels: pd.Series, persist: int = 2) -> pd.Series:
        """
        Applique une règle de confirmation pour éviter le whipsaw.
        
        Un changement de régime n'est appliqué que si le nouveau régime
        persiste pendant `persist` mois consécutifs.
        """
        held = []
        current = None
        
        for i, (dt, lab) in enumerate(labels.items()):
            if i + 1 >= persist:
                last_k = labels.iloc[i - persist + 1:i + 1]
                if (last_k.nunique() == 1) and (last_k.iloc[-1] != current):
                    current = lab
            held.append(current)
        
        return pd.Series(held, index=labels.index, name="label_held")
    
    @staticmethod
    def label_to_weights(label: str) -> Tuple[float, float]:
        """
        Convertit un label de régime en poids d'allocation.
        
        Returns:
            Tuple[weight_stocks, weight_bonds]
        """
        if label == "equities":
            return 1.0, 0.0
        elif label == "rates":
            return 0.0, 1.0
        elif label == "both":
            return 0.6, 0.4  # 60/40 balanced
        else:  # "none" -> cash
            return 0.0, 0.0
    
    def run_backtest(self, labels_held: pd.Series, y: pd.DataFrame) -> pd.DataFrame:
        """
        Exécute le backtest de la stratégie.
        
        Returns:
            DataFrame avec rendements et allocations
        """
        common = labels_held.index.intersection(y.index)
        L = labels_held.loc[common]
        Y = y.loc[common]
        
        w = np.array([self.label_to_weights(lab) for lab in L])
        
        strat_r = w[:, 0] * Y["spx_next"].values + w[:, 1] * Y["vbmfx_next"].values
        ret_60_40 = 0.6 * Y["spx_next"].values + 0.4 * Y["vbmfx_next"].values
        
        out = pd.DataFrame({
            "ret_strategy": strat_r,
            "ret_spx": Y["spx_next"].values,
            "ret_bond": Y["vbmfx_next"].values,
            "ret_60_40": ret_60_40,
            "w_spx": w[:, 0],
            "w_bond": w[:, 1],
            "label": L.values,
        }, index=common)
        
        return out
    
    @staticmethod
    def perf_stats(simple_returns: pd.Series, ann_factor: int = 12) -> Dict:
        """
        Calcule les statistiques de performance.
        
        Returns:
            Dict avec N, CAGR, Vol, Sharpe, MaxDD
        """
        r = simple_returns.fillna(0.0)
        n = len(r)
        
        if n == 0:
            return {"N": 0}
        
        cagr = (1.0 + r).prod() ** (ann_factor / n) - 1.0
        vol = r.std(ddof=0) * (ann_factor ** 0.5)
        sharpe = cagr / vol if vol > 0 else np.nan
        
        curve = (1.0 + r).cumprod()
        peak = curve.cummax()
        mdd = ((curve / peak) - 1.0).min()
        
        return {
            "N": n,
            "CAGR": cagr,
            "Vol": vol,
            "Sharpe": sharpe,
            "MaxDD": mdd,
        }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

class Visualizer:
    """Génération des visualisations."""
    
    LABEL_COLORS = {
        "equities": "#2563eb",  # Blue
        "rates": "#22c55e",     # Green
        "both": "#f59e0b",      # Amber
        "none": "#64748b",      # Gray
    }
    
    LABEL_NAMES = {
        "equities": "Actions",
        "rates": "Obligations",
        "both": "60/40",
        "none": "Cash",
    }
    
    @staticmethod
    def plot_cumulative_performance(bt: pd.DataFrame, title: str = "Performance Cumulée"):
        """Plot des courbes de performance cumulée."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        def cumcurve(r):
            return (1.0 + r.fillna(0.0)).cumprod()
        
        curves = pd.DataFrame({
            "Strategy": cumcurve(bt["ret_strategy"]),
            "S&P 500": cumcurve(bt["ret_spx"]),
            "Bonds": cumcurve(bt["ret_bond"]),
            "60/40": cumcurve(bt["ret_60_40"]),
        })
        
        curves.plot(ax=ax, linewidth=2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_regime_timeline(labels_held: pd.Series, title: str = "Régimes Détectés"):
        """Plot de la timeline des régimes."""
        fig, ax = plt.subplots(figsize=(14, 3))
        
        encoding = {"none": 0, "rates": 1, "both": 2, "equities": 3}
        code = labels_held.fillna("none").map(encoding)
        
        ax.fill_between(code.index, 0, code.values, step="post", alpha=0.7)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["Cash", "Obligations", "60/40", "Actions"])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_regime_comparison(labels_hmm: pd.Series, labels_kmeans: pd.Series):
        """Comparaison des régimes HMM vs K-Means."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
        
        encoding = {"none": 0, "rates": 1, "both": 2, "equities": 3}
        
        for ax, (labels, name) in zip(axes, [(labels_hmm, "HMM"), (labels_kmeans, "K-Means")]):
            code = labels.fillna("none").map(encoding)
            ax.fill_between(code.index, 0, code.values, step="post", alpha=0.7)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(["Cash", "Oblig.", "60/40", "Actions"])
            ax.set_title(f"Régimes - {name}", fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_performance_by_regime(bt: pd.DataFrame):
        """Performance des actifs par régime."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        grouped = bt.groupby("label").agg({
            "ret_spx": ["mean", "std"],
            "ret_bond": ["mean", "std"],
        })
        
        labels = grouped.index.tolist()
        x = np.arange(len(labels))
        width = 0.35
        
        spx_means = grouped[("ret_spx", "mean")] * 12 * 100  # Annualisé en %
        bond_means = grouped[("ret_bond", "mean")] * 12 * 100
        
        ax.bar(x - width/2, spx_means, width, label='Actions', color='#2563eb')
        ax.bar(x + width/2, bond_means, width, label='Obligations', color='#22c55e')
        
        ax.set_ylabel('Rendement Annualisé (%)')
        ax.set_title('Performance par Régime', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([Visualizer.LABEL_NAMES.get(l, l) for l in labels])
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig


# ==============================================================================
# EXPORT FOR DASHBOARD
# ==============================================================================

def export_for_dashboard(X: pd.DataFrame, y: pd.DataFrame,
                         hmm_labels: pd.Series, kmeans_labels: pd.Series,
                         bt_hmm: pd.DataFrame, bt_kmeans: pd.DataFrame,
                         output_path: str = "dashboard_data.json"):
    """
    Exporte les données au format JSON pour le dashboard React.
    """
    import json
    
    # Préparer les données
    common_idx = hmm_labels.index.intersection(kmeans_labels.index)
    
    data = {
        "metadata": {
            "start_date": str(common_idx.min().date()),
            "end_date": str(common_idx.max().date()),
            "n_months": len(common_idx),
            "generated_at": datetime.now().isoformat(),
        },
        "current_regime": {
            "hmm": hmm_labels.iloc[-1] if len(hmm_labels) > 0 else "none",
            "kmeans": kmeans_labels.iloc[-1] if len(kmeans_labels) > 0 else "none",
            "date": str(common_idx[-1].date()) if len(common_idx) > 0 else None,
        },
        "time_series": [],
        "performance_stats": {},
        "regime_distribution": {},
    }
    
    # Time series data
    for idx in common_idx:
        point = {
            "date": str(idx.date()),
            "hmm_regime": hmm_labels.loc[idx] if idx in hmm_labels.index else None,
            "kmeans_regime": kmeans_labels.loc[idx] if idx in kmeans_labels.index else None,
        }
        
        # Features
        if idx in X.index:
            for col in X.columns:
                val = X.loc[idx, col]
                point[f"feature_{col}"] = float(val) if pd.notna(val) else None
        
        # Returns
        if idx in y.index:
            point["ret_spx"] = float(y.loc[idx, "spx_next"]) if pd.notna(y.loc[idx, "spx_next"]) else None
            point["ret_bond"] = float(y.loc[idx, "vbmfx_next"]) if pd.notna(y.loc[idx, "vbmfx_next"]) else None
        
        data["time_series"].append(point)
    
    # Performance stats
    backtest = BacktestEngine()
    for name, bt in [("hmm", bt_hmm), ("kmeans", bt_kmeans)]:
        data["performance_stats"][name] = {
            "strategy": backtest.perf_stats(bt["ret_strategy"]),
            "spx": backtest.perf_stats(bt["ret_spx"]),
            "bonds": backtest.perf_stats(bt["ret_bond"]),
            "balanced_60_40": backtest.perf_stats(bt["ret_60_40"]),
        }
    
    # Regime distribution
    for name, labels in [("hmm", hmm_labels), ("kmeans", kmeans_labels)]:
        counts = labels.value_counts(normalize=True)
        data["regime_distribution"][name] = {k: float(v) for k, v in counts.items()}
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"\n✓ Données exportées vers {output_path}")
    return data


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Point d'entrée principal."""
    
    print("=" * 70)
    print("MACRO REGIME DETECTION TOOL - K-MEANS vs HMM")
    print("=" * 70)
    
    # Configuration
    config = Config()
    
    # 1. Chargement des données
    loader = MacroDataLoader()
    macro, assets = loader.load_all_data(start=config.START_DATE)
    X, y = loader.prepare_features(macro, assets, config)
    
    # 2. HMM Detection
    print("\n" + "=" * 50)
    print("HMM REGIME DETECTION")
    print("=" * 50)
    
    hmm_config = HMMConfig(
        n_states=config.N_STATES_HMM,
        cov_type=config.HMM_COV_TYPE,
        n_iter=config.HMM_N_ITER,
        n_init=config.HMM_N_INIT,
        random_state=config.RANDOM_STATE
    )
    
    hmm_detector = HMMRegimeDetector(hmm_config)
    hmm_labels_pred, hmm_pmax, hmm_probas = hmm_detector.rolling_predict(
        X, y, window=config.WINDOW_MONTHS
    )
    
    # 3. K-Means Detection
    print("\n" + "=" * 50)
    print("K-MEANS REGIME DETECTION")
    print("=" * 50)
    
    kmeans_detector = KMeansRegimeDetector(
        n_clusters=config.N_CLUSTERS_KMEANS,
        n_init=config.KMEANS_N_INIT,
        max_iter=config.KMEANS_MAX_ITER,
        random_state=config.RANDOM_STATE
    )
    
    kmeans_labels_pred, kmeans_metrics = kmeans_detector.rolling_predict(
        X, y, window=config.WINDOW_MONTHS
    )
    
    # 4. Apply confirmation rule
    backtest = BacktestEngine()
    
    hmm_labels_held = backtest.apply_confirmation(hmm_labels_pred, config.PERSISTENCE)
    kmeans_labels_held = backtest.apply_confirmation(kmeans_labels_pred, config.PERSISTENCE)
    
    # 5. Backtest
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    
    bt_hmm = backtest.run_backtest(hmm_labels_held, y)
    bt_kmeans = backtest.run_backtest(kmeans_labels_held, y)
    
    # 6. Performance Stats
    print("\n--- HMM Strategy ---")
    stats_hmm = pd.DataFrame({
        "Strategy": backtest.perf_stats(bt_hmm["ret_strategy"]),
        "S&P 500": backtest.perf_stats(bt_hmm["ret_spx"]),
        "Bonds": backtest.perf_stats(bt_hmm["ret_bond"]),
        "60/40": backtest.perf_stats(bt_hmm["ret_60_40"]),
    }).T
    print(stats_hmm.round(4))
    
    print("\n--- K-Means Strategy ---")
    stats_kmeans = pd.DataFrame({
        "Strategy": backtest.perf_stats(bt_kmeans["ret_strategy"]),
        "S&P 500": backtest.perf_stats(bt_kmeans["ret_spx"]),
        "Bonds": backtest.perf_stats(bt_kmeans["ret_bond"]),
        "60/40": backtest.perf_stats(bt_kmeans["ret_60_40"]),
    }).T
    print(stats_kmeans.round(4))
    
    # 7. Current Regime
    print("\n" + "=" * 50)
    print("CURRENT REGIME & ALLOCATION")
    print("=" * 50)
    
    current_date = hmm_labels_held.index[-1]
    print(f"\nDate: {current_date.date()}")
    print(f"HMM Regime: {hmm_labels_held.iloc[-1]}")
    print(f"K-Means Regime: {kmeans_labels_held.iloc[-1]}")
    
    hmm_w = backtest.label_to_weights(hmm_labels_held.iloc[-1])
    kmeans_w = backtest.label_to_weights(kmeans_labels_held.iloc[-1])
    
    print(f"\nHMM Allocation: {hmm_w[0]*100:.0f}% Actions / {hmm_w[1]*100:.0f}% Bonds")
    print(f"K-Means Allocation: {kmeans_w[0]*100:.0f}% Actions / {kmeans_w[1]*100:.0f}% Bonds")
    
    # 8. Concordance
    common_idx = hmm_labels_held.index.intersection(kmeans_labels_held.index)
    agreement = (hmm_labels_held.loc[common_idx] == kmeans_labels_held.loc[common_idx]).mean()
    print(f"\nConcordance HMM/K-Means: {agreement*100:.1f}%")
    
    # 9. Export for dashboard
    export_for_dashboard(X, y, hmm_labels_held, kmeans_labels_held, bt_hmm, bt_kmeans)
    
    # 10. Generate plots
    print("\n" + "=" * 50)
    print("GENERATING PLOTS...")
    print("=" * 50)
    
    viz = Visualizer()
    
    fig1 = viz.plot_cumulative_performance(bt_hmm, "Performance Cumulée - HMM Strategy")
    fig1.savefig("performance_hmm.png", dpi=150, bbox_inches='tight')
    
    fig2 = viz.plot_cumulative_performance(bt_kmeans, "Performance Cumulée - K-Means Strategy")
    fig2.savefig("performance_kmeans.png", dpi=150, bbox_inches='tight')
    
    fig3 = viz.plot_regime_comparison(hmm_labels_held, kmeans_labels_held)
    fig3.savefig("regime_comparison.png", dpi=150, bbox_inches='tight')
    
    fig4 = viz.plot_performance_by_regime(bt_hmm)
    fig4.savefig("performance_by_regime.png", dpi=150, bbox_inches='tight')
    
    print("\n✓ Plots sauvegardés")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    return {
        "X": X,
        "y": y,
        "hmm_labels": hmm_labels_held,
        "kmeans_labels": kmeans_labels_held,
        "bt_hmm": bt_hmm,
        "bt_kmeans": bt_kmeans,
        "stats_hmm": stats_hmm,
        "stats_kmeans": stats_kmeans,
    }


if __name__ == "__main__":
    results = main()
