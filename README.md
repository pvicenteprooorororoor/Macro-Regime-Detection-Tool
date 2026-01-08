# 🎯 Macro Regime Detection Tool

**Outil de détection de régimes macroéconomiques avec comparaison K-Means vs HMM**

## 📋 Description

Cet outil identifie les régimes macroéconomiques (Expansion, Récession, etc.) à partir de données réelles et recommande une allocation d'actifs optimale entre Actions et Obligations.

### Caractéristiques :
- ✅ **Pas de look-ahead bias** : Estimation sur fenêtre glissante uniquement
- ✅ **Deux méthodes comparées** : K-Means Clustering vs Hidden Markov Model
- ✅ **Données réelles** : FRED (macro) + Yahoo Finance (actifs)
- ✅ **Dashboard interactif** : Visualisation des résultats
- ✅ **Backtest complet** : Performance historique de la stratégie

---

## 🚀 Installation Rapide

### Prérequis
- Python 3.9+ 
- Une clé API FRED (gratuite) : https://fred.stlouisfed.org/docs/api/api_key.html

### Étapes

```bash
# 1. Créer un dossier et y copier les fichiers
mkdir macro_regime_tool
cd macro_regime_tool

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv

# Sur Windows:
venv\Scripts\activate

# Sur Mac/Linux:
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API FRED
# Créer un fichier .env avec votre clé :
echo "FRED_API_KEY=votre_cle_api_ici" > .env

# 5. Lancer l'analyse
python regime_detector.py

# 6. (Optionnel) Lancer le dashboard Streamlit
streamlit run dashboard_streamlit.py
```

---

## 📁 Structure des Fichiers

```
macro_regime_tool/
│
├── regime_detector.py      # Code principal (K-Means + HMM)
├── dashboard_streamlit.py  # Dashboard interactif (Streamlit)
├── requirements.txt        # Dépendances Python
├── .env                    # Votre clé API FRED (à créer)
├── README.md              # Ce fichier
│
└── outputs/               # Résultats générés
    ├── dashboard_data.json
    ├── performance_hmm.png
    ├── performance_kmeans.png
    └── regime_comparison.png
```

---

## 📊 Variables Macroéconomiques Utilisées

| Variable | Source FRED | Justification Économique |
|----------|-------------|-------------------------|
| Chômage (UNRATE) | FRED | Indicateur retardé du cycle |
| Inflation (CPI) | FRED | Politique monétaire, taux réels |
| Taux 10 ans (GS10) | FRED | Anticipations croissance/inflation |
| Taux 2 ans (GS2) | FRED | Anticipations Fed |
| VIX | FRED | Aversion au risque |
| Spread BAA | FRED | Stress crédit corporate |

### Features dérivées :
- `infl_mom` : Momentum d'inflation (variation log CPI)
- `2s10s_spread` : Pente de courbe (10Y - 2Y)
- `ust10y_d`, `ust2y_d` : Variations mensuelles des taux

---

## 🎯 Régimes et Allocations

| Régime | Condition | Allocation |
|--------|-----------|------------|
| **Equities** | Actions performantes, bonds faibles | 100% Actions |
| **Rates** | Bonds performants, actions faibles | 100% Obligations |
| **Both** | Les deux classes performantes | 60% Actions / 40% Bonds |
| **None** | Les deux faibles | 100% Cash |

---

## 🔬 Méthodologie

### K-Means Clustering
- Partitionne les observations en K clusters
- Minimise la variance intra-cluster
- **Avantage** : Simple, interprétable
- **Limite** : Pas de structure temporelle

### Hidden Markov Model (HMM)
- États cachés évoluant selon une chaîne de Markov
- Distributions d'émission gaussiennes
- **Avantage** : Capture la persistance des régimes
- **Limite** : Plus complexe, peut être instable

### Contraintes anti-look-ahead :
1. Features shiftées de 1 mois (X_t utilise info jusqu'à t-1)
2. Entraînement sur fenêtre [t-window, t-1] uniquement
3. Prédiction à t via posterior sur [t-window, t]
4. Règle de confirmation de 2 mois avant changement

---

## 📈 Résultats Attendus

Après exécution, vous obtiendrez :

1. **Statistiques de performance** :
   - CAGR, Volatilité, Sharpe, Max Drawdown
   - Comparaison Stratégie vs Buy&Hold vs 60/40

2. **Régime actuel** :
   - Détection du régime en cours
   - Allocation recommandée

3. **Visualisations** :
   - Courbes de performance cumulée
   - Timeline des régimes
   - Performance par régime

---

## ⚠️ Limitations et Avertissements

1. **Retard de détection** : Les changements de régime sont détectés avec retard
2. **Données révisées** : Les données macro sont souvent révisées après publication
3. **Passé ≠ Futur** : La performance passée ne garantit pas les résultats futurs
4. **Cet outil est éducatif** : Ne constitue pas un conseil en investissement

---

## 🔧 Personnalisation

### Modifier le nombre de régimes
```python
# Dans regime_detector.py
config = Config()
config.N_STATES_HMM = 5      # HMM: 5 états
config.N_CLUSTERS_KMEANS = 4  # K-Means: 4 clusters
```

### Modifier la fenêtre rolling
```python
config.WINDOW_YEARS = 20  # 20 ans de données pour l'entraînement
```

### Modifier la règle de confirmation
```python
config.PERSISTENCE = 2  # 2 mois de confirmation avant changement
```

---

## 📞 Support

Pour toute question sur le code ou la méthodologie, consultez les commentaires détaillés dans `regime_detector.py`.

---

## 📜 Licence

Usage personnel et éducatif uniquement.
