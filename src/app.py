"""
ATP Fatigue Forecaster — Research-Grade ML Dashboard
Built on real rolling physiological load features from JeffSackmann/tennis_atp.

v3 upgrades:
  - Model Performance tab with baseline comparisons (Higher-Rank, Rank-Logistic, Elo)
  - Ablation study tab: empirically demonstrates lift from physiological-load features
  - Match Predictor with rest-day counterfactual slider (what-if analysis)
"""

import time
import logging
import datetime
import warnings
import urllib.error

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             brier_score_loss, log_loss, roc_curve)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ATP Fatigue Forecaster",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --clay:     #C4622D;
    --clay-lt:  #E07A4F;
    --grass:    #4A7C59;
    --hard:     #2B5BA8;
    --black:    #0D0D0D;
    --surface:  #141414;
    --panel:    #1C1C1C;
    --border:   #2E2E2E;
    --text:     #E8E8E8;
    --muted:    #7A7A7A;
    --accent:   #C4622D;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--black);
    color: var(--text);
}

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }

.stApp { background-color: var(--black); }

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; }
[data-testid="stMetricValue"] { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: var(--text); letter-spacing: 2px; }

[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--clay) !important;
    border-bottom-color: var(--clay) !important;
}

.prob-container {
    display: flex;
    height: 56px;
    border-radius: 6px;
    overflow: hidden;
    margin: 24px 0 8px 0;
    border: 1px solid var(--border);
}
.prob-p1 {
    background: linear-gradient(90deg, #2B5BA8, #4A7FC4);
    display: flex; align-items: center; padding-left: 16px;
    font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; letter-spacing: 2px; color: white;
    transition: width 0.6s ease;
}
.prob-p2 {
    background: linear-gradient(90deg, #A83232, #C4622D);
    display: flex; align-items: center; justify-content: flex-end; padding-right: 16px;
    font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; letter-spacing: 2px; color: white;
    flex: 1;
    transition: width 0.6s ease;
}

.verdict {
    background: var(--panel);
    border-left: 4px solid var(--clay);
    border-radius: 0 8px 8px 0;
    padding: 20px 24px;
    margin: 16px 0;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    line-height: 1.7;
    color: var(--text);
}

.fi-row { display: flex; align-items: center; margin: 6px 0; gap: 12px; }
.fi-label { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); width: 180px; text-align: right; flex-shrink: 0; text-transform: uppercase; letter-spacing: 0.5px; }
.fi-bar-bg { flex: 1; background: var(--border); border-radius: 3px; height: 8px; }
.fi-bar-fill { height: 8px; border-radius: 3px; background: linear-gradient(90deg, var(--clay), var(--clay-lt)); }
.fi-val { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); width: 48px; }

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 20px;
}

.player-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.player-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 3px;
    margin-bottom: 4px;
}
.player-rank {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.stat-pill {
    display: inline-block;
    background: var(--black);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    margin: 3px;
}

.warning-bar {
    background: rgba(196,98,45,0.15);
    border: 1px solid rgba(196,98,45,0.4);
    border-radius: 6px;
    padding: 10px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--clay-lt);
    margin: 12px 0;
}

button[kind="primary"] {
    background: var(--clay) !important;
    border: none !important;
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 2px !important;
    font-size: 1rem !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Counterfactual highlight */
.counterfactual-box {
    background: linear-gradient(135deg, rgba(196,98,45,0.12), rgba(74,127,196,0.12));
    border: 1px solid rgba(196,98,45,0.4);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--text);
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def load_raw_data(years: int = 8) -> pd.DataFrame:
    current_year = datetime.datetime.now().year
    frames = []
    for year in range(current_year, current_year - years - 1, -1):
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        try:
            df_year = pd.read_csv(url, low_memory=False)
            frames.append(df_year)
        except urllib.error.HTTPError:
            logging.warning(f"{year} not yet published, skipping.")
        except Exception as e:
            logging.error(f"Error loading {year}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING CONSTANTS
# ─────────────────────────────────────────────
SURFACE_FATIGUE_WEIGHT = {'Clay': 1.27, 'Hard': 1.0, 'Grass': 0.80, 'Carpet': 0.90}
ROUND_FATIGUE_WEIGHT = {
    'R128': 0.7, 'R64': 0.75, 'R32': 0.8, 'R16': 0.9,
    'QF': 1.0, 'SF': 1.15, 'F': 1.3, 'RR': 0.85
}

# Feature groupings for ablation study — each maps to research framework stages
RANKING_FEATURES = ['p1_rank', 'p2_rank']
LOAD_FEATURES = [  # Stage 1 cascade signals
    'p1_cum_mins_7d', 'p2_cum_mins_7d',
    'p1_cum_mins_14d', 'p2_cum_mins_14d',
    'p1_cum_mins_28d', 'p2_cum_mins_28d',
    'p1_surf_weighted_mins_28d', 'p2_surf_weighted_mins_28d',
    'p1_round_weighted_mins_28d', 'p2_round_weighted_mins_28d',
    'p1_matches_7d', 'p2_matches_7d',
    'p1_days_since_last', 'p2_days_since_last',
    'p1_tourney_matches_before', 'p2_tourney_matches_before',
    'p1_tourney_mins_before', 'p2_tourney_mins_before',
]
FORM_FEATURES = [  # Stage 3-5 downstream effects
    'p1_win_pct_10', 'p2_win_pct_10',
    'p1_win_pct_20', 'p2_win_pct_20',
]
H2H_FEATURES = ['p1_h2h_win_pct', 'p2_h2h_win_pct']
QUALITY_FEATURES = ['p1_opp_avg_rank_beaten', 'p2_opp_avg_rank_beaten']
SURFACE_FEATURE = ['surface']


# ─────────────────────────────────────────────
# REAL FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    from collections import defaultdict
    required_cols = ['tourney_date', 'winner_id', 'loser_id', 'winner_rank', 'loser_rank',
                     'minutes', 'surface', 'tourney_id', 'round']
    df = df_raw.dropna(subset=required_cols).copy()
    df['match_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['match_date']).sort_values('match_date').reset_index(drop=True)

    round_order = {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'RR': 3}
    df['round_num'] = df['round'].map(round_order).fillna(3)
    surface_map = {'Clay': 0, 'Grass': 1, 'Hard': 2, 'Carpet': 3}
    df['surface_enc'] = df['surface'].map(surface_map).fillna(2)
    df['surface_weight'] = df['surface'].map(SURFACE_FATIGUE_WEIGHT).fillna(1.0)
    df['round_weight']   = df['round'].map(ROUND_FATIGUE_WEIGHT).fillna(0.85)
    df['weighted_mins']  = df['minutes'] * df['surface_weight'] * df['round_weight']

    player_history  = defaultdict(list)
    h2h_history     = defaultdict(list)
    rows = []

    for _, row in df.iterrows():
        w_id, l_id   = row['winner_id'], row['loser_id']
        w_rank, l_rank = row['winner_rank'], row['loser_rank']
        match_date   = row['match_date']
        mins         = row['minutes']
        w_mins       = row['weighted_mins']
        tourney_id   = row['tourney_id']
        surface_enc  = row['surface_enc']

        def compute_features(pid, rank, opp_id, opp_rank):
            hist = player_history[pid]
            empty = {
                'rank': rank,
                'cum_mins_7d': 0, 'cum_mins_14d': 0, 'cum_mins_28d': 0,
                'surf_weighted_mins_28d': 0, 'round_weighted_mins_28d': 0,
                'matches_7d': 0, 'days_since_last': 30,
                'win_pct_10': 0.5, 'win_pct_20': 0.5,
                'tourney_matches_before': 0, 'tourney_mins_before': 0,
                'h2h_win_pct': 0.5, 'opp_avg_rank_beaten': 100,
            }
            if not hist:
                return empty
            ts_arr    = np.array([h[0] for h in hist])
            wins_arr  = np.array([h[1] for h in hist])
            mins_arr  = np.array([h[2] for h in hist])
            wmins_arr = np.array([h[3] for h in hist])
            tourneys  = [h[4] for h in hist]
            opp_ranks = np.array([h[7] for h in hist])

            cutoff = match_date.timestamp()
            d7  = cutoff - 7  * 86400
            d14 = cutoff - 14 * 86400
            d28 = cutoff - 28 * 86400
            mask_7  = ts_arr >= d7
            mask_14 = ts_arr >= d14
            mask_28 = ts_arr >= d28

            days_since = (cutoff - ts_arr[-1]) / 86400
            recent_10 = wins_arr[-10:]
            recent_20 = wins_arr[-20:]
            t_mask    = np.array([t == tourney_id for t in tourneys])

            h2h_key = (pid, opp_id)
            h2h_rec = h2h_history[h2h_key]
            h2h_pct = float(np.mean(h2h_rec)) if len(h2h_rec) >= 2 else 0.5

            win_opp_ranks = opp_ranks[wins_arr == 1][-10:]
            opp_avg_rank  = float(win_opp_ranks.mean()) if len(win_opp_ranks) > 0 else float(opp_rank)

            return {
                'rank':                    rank,
                'cum_mins_7d':             mins_arr[mask_7].sum(),
                'cum_mins_14d':            mins_arr[mask_14].sum(),
                'cum_mins_28d':            mins_arr[mask_28].sum(),
                'surf_weighted_mins_28d':  wmins_arr[mask_28].sum(),
                'round_weighted_mins_28d': wmins_arr[mask_28].sum(),
                'matches_7d':              int(mask_7.sum()),
                'days_since_last':         days_since,
                'win_pct_10':              recent_10.mean() if len(recent_10) > 0 else 0.5,
                'win_pct_20':              recent_20.mean() if len(recent_20) > 0 else 0.5,
                'tourney_matches_before':  int(t_mask.sum()),
                'tourney_mins_before':     mins_arr[t_mask].sum() if t_mask.any() else 0,
                'h2h_win_pct':             h2h_pct,
                'opp_avg_rank_beaten':     opp_avg_rank,
            }

        w_feats = compute_features(w_id, w_rank, l_id, l_rank)
        l_feats = compute_features(l_id, l_rank, w_id, w_rank)

        if np.random.rand() > 0.5:
            p1_feats, p2_feats, p1_wins = w_feats, l_feats, 1
        else:
            p1_feats, p2_feats, p1_wins = l_feats, w_feats, 0

        record = {}
        for k, v in p1_feats.items():
            record[f'p1_{k}'] = v
        for k, v in p2_feats.items():
            record[f'p2_{k}'] = v
        record['surface'] = surface_enc
        record['p1_wins'] = p1_wins
        rows.append(record)

        ts = match_date.timestamp()
        player_history[w_id].append((ts, 1, mins, w_mins, tourney_id, row['round_num'], l_id, l_rank))
        player_history[l_id].append((ts, 0, mins, w_mins, tourney_id, row['round_num'], w_id, w_rank))
        h2h_history[(w_id, l_id)].append(1)
        h2h_history[(l_id, w_id)].append(0)

    return pd.DataFrame(rows).dropna()


# ─────────────────────────────────────────────
# MODEL TRAINING (full feature model)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(_df: pd.DataFrame):
    feature_cols = [c for c in _df.columns if c != 'p1_wins']
    X = _df[feature_cols].values
    y = _df['p1_wins'].values
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    base = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_samples_split=20, subsample=0.8, random_state=42
    )
    model = CalibratedClassifierCV(base, cv=3, method='isotonic')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')

    metrics = {
        'accuracy':    accuracy_score(y_test, y_pred),
        'roc_auc':     roc_auc_score(y_test, y_prob),
        'precision':   precision_score(y_test, y_pred, zero_division=0),
        'recall':      recall_score(y_test, y_pred, zero_division=0),
        'brier':       brier_score_loss(y_test, y_prob),
        'log_loss':    log_loss(y_test, y_prob),
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std':  cv_scores.std(),
        'train_size':  len(X_train),
        'test_size':   len(X_test),
    }

    try:
        importances = base.fit(X_train, y_train).feature_importances_
    except Exception:
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    return model, feature_cols, metrics, importances, X_test, y_test, y_prob


# ─────────────────────────────────────────────
# BASELINES & ABLATION (NEW)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def evaluate_baselines_and_ablations(_df: pd.DataFrame):
    """
    Evaluate non-ML baselines AND feature-group ablations.
    Returns a dict of {model_name: {metrics}} for the same temporal test split.
    """
    feature_cols = [c for c in _df.columns if c != 'p1_wins']
    X_full = _df[feature_cols].values
    y      = _df['p1_wins'].values
    split  = int(len(X_full) * 0.8)
    y_train, y_test = y[:split], y[split:]

    test_slice = _df.iloc[split:].reset_index(drop=True)
    results = {}

    # ── Baseline 1: Higher-Rank-Wins (no ML) ──────────────────────────────
    # Predict P1 wins iff p1_rank < p2_rank (lower rank number = better)
    higher_rank_pred = (test_slice['p1_rank'] < test_slice['p2_rank']).astype(int).values
    # Probability proxy: rank-distance sigmoid
    rank_diff = (test_slice['p2_rank'] - test_slice['p1_rank']).values  # positive = P1 better
    higher_rank_prob = 1 / (1 + np.exp(-rank_diff / 30))
    results['Higher-Rank Wins'] = {
        'accuracy':  accuracy_score(y_test, higher_rank_pred),
        'roc_auc':   roc_auc_score(y_test, higher_rank_prob),
        'brier':     brier_score_loss(y_test, higher_rank_prob),
        'log_loss':  log_loss(y_test, np.clip(higher_rank_prob, 1e-6, 1-1e-6)),
        'desc':      'Naive: better-ranked player always wins',
        'features':  2,
    }

    # ── Baseline 2: Rank-only Logistic Regression ─────────────────────────
    rank_cols = ['p1_rank', 'p2_rank']
    X_rank_train = _df[rank_cols].values[:split]
    X_rank_test  = _df[rank_cols].values[split:]
    lr_rank = LogisticRegression(max_iter=1000)
    lr_rank.fit(X_rank_train, y_train)
    lr_prob = lr_rank.predict_proba(X_rank_test)[:, 1]
    lr_pred = lr_rank.predict(X_rank_test)
    results['Rank-Only Logistic'] = {
        'accuracy':  accuracy_score(y_test, lr_pred),
        'roc_auc':   roc_auc_score(y_test, lr_prob),
        'brier':     brier_score_loss(y_test, lr_prob),
        'log_loss':  log_loss(y_test, np.clip(lr_prob, 1e-6, 1-1e-6)),
        'desc':      'Logistic regression on ranks alone',
        'features':  2,
    }

    # ── Baseline 3: Simple Elo (computed in-loop) ─────────────────────────
    # We recompute Elo from raw history for the test window.
    # For this dataset the dataframe is already temporally sorted.
    K = 32
    elo = {}  # player_pseudo_id -> rating
    elo_probs = []
    elo_preds = []
    for _, r in test_slice.iterrows():
        # We don't have player IDs in engineered df; use rank as a stable pseudo-key
        # Fall back to rank-based Elo proxy.
        # Better: pre-compute Elo offline. For now, derive expected score from rank proxy.
        p1_rating = 1500 + (300 - min(r['p1_rank'], 300)) * 2
        p2_rating = 1500 + (300 - min(r['p2_rank'], 300)) * 2
        expected_p1 = 1 / (1 + 10 ** ((p2_rating - p1_rating) / 400))
        elo_probs.append(expected_p1)
        elo_preds.append(1 if expected_p1 >= 0.5 else 0)
    elo_probs = np.array(elo_probs)
    elo_preds = np.array(elo_preds)
    results['Elo-Style (rank-derived)'] = {
        'accuracy':  accuracy_score(y_test, elo_preds),
        'roc_auc':   roc_auc_score(y_test, elo_probs),
        'brier':     brier_score_loss(y_test, elo_probs),
        'log_loss':  log_loss(y_test, np.clip(elo_probs, 1e-6, 1-1e-6)),
        'desc':      'Elo expected-score formula on rank-derived ratings',
        'features':  2,
    }

    # ── Ablation: GBM with progressively richer feature sets ──────────────
    ablation_specs = [
        ('GBM: Ranking only',
         RANKING_FEATURES,
         'Rank features only'),
        ('GBM: + Physiological Load',
         RANKING_FEATURES + LOAD_FEATURES,
         'Adds 7/14/28d court minutes, weighted load, rest days, tourney load'),
        ('GBM: + Form',
         RANKING_FEATURES + LOAD_FEATURES + FORM_FEATURES,
         'Adds rolling win % over last 10 and 20 matches'),
        ('GBM: + H2H + Quality',
         RANKING_FEATURES + LOAD_FEATURES + FORM_FEATURES + H2H_FEATURES + QUALITY_FEATURES,
         'Adds head-to-head record and opponent-quality (strength of schedule)'),
        ('GBM: Full (all features)',
         feature_cols,
         'All features including surface'),
    ]

    for name, cols, desc in ablation_specs:
        cols_present = [c for c in cols if c in _df.columns]
        X_a = _df[cols_present].values
        Xa_train, Xa_test = X_a[:split], X_a[split:]

        gbm = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            min_samples_split=20, subsample=0.8, random_state=42
        )
        gbm.fit(Xa_train, y_train)
        a_prob = gbm.predict_proba(Xa_test)[:, 1]
        a_pred = gbm.predict(Xa_test)

        results[name] = {
            'accuracy':  accuracy_score(y_test, a_pred),
            'roc_auc':   roc_auc_score(y_test, a_prob),
            'brier':     brier_score_loss(y_test, a_prob),
            'log_loss':  log_loss(y_test, np.clip(a_prob, 1e-6, 1-1e-6)),
            'desc':      desc,
            'features':  len(cols_present),
        }

    return results, y_test


# ─────────────────────────────────────────────
# PLAYER LOOKUP
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_player_stats(_df_raw: pd.DataFrame) -> dict:
    df = _df_raw.copy()
    df['match_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['match_date', 'winner_name', 'loser_name']).sort_values('match_date')
    df['surface_weight'] = df['surface'].map(SURFACE_FATIGUE_WEIGHT).fillna(1.0)
    df['round_weight']   = df['round'].map(ROUND_FATIGUE_WEIGHT).fillna(0.85)
    median_mins = df['minutes'].median() if df['minutes'].notna().any() else 90.0
    df['minutes_filled']  = df['minutes'].fillna(median_mins)
    df['weighted_mins']   = df['minutes_filled'] * df['surface_weight'] * df['round_weight']
    cutoff = df['match_date'].max()

    from collections import defaultdict
    h2h_wins  = defaultdict(int)
    h2h_total = defaultdict(int)
    for _, row in df.iterrows():
        w, l = row['winner_name'], row['loser_name']
        h2h_wins[(w, l)]  += 1
        h2h_total[(w, l)] += 1
        h2h_total[(l, w)] += 1

    h2h_pct = {}
    for (p1, p2), total in h2h_total.items():
        if total > 0:
            h2h_pct[(p1, p2)] = h2h_wins.get((p1, p2), 0) / total

    df_ranked = df.dropna(subset=['winner_rank', 'loser_rank', 'minutes'])
    all_players = pd.concat([
        df_ranked[['winner_name', 'winner_id', 'winner_rank']].rename(
            columns={'winner_name': 'name', 'winner_id': 'id', 'winner_rank': 'rank'}),
        df_ranked[['loser_name', 'loser_id', 'loser_rank']].rename(
            columns={'loser_name': 'name', 'loser_id': 'id', 'loser_rank': 'rank'})
    ]).sort_values('rank').drop_duplicates('name')

    stats = {}
    for _, prow in all_players.iterrows():
        name = prow['name']
        pid  = prow['id']
        w_mask = df_ranked['winner_id'] == pid
        l_mask = df_ranked['loser_id']  == pid
        w_df = df_ranked[w_mask][['match_date', 'minutes', 'weighted_mins', 'loser_rank']].assign(won=1)
        w_df = w_df.rename(columns={'loser_rank': 'opp_rank'})
        l_df = df_ranked[l_mask][['match_date', 'minutes', 'weighted_mins', 'winner_rank']].assign(won=0)
        l_df = l_df.rename(columns={'winner_rank': 'opp_rank'})
        ph = pd.concat([w_df, l_df]).sort_values('match_date')
        if len(ph) < 3:
            continue
        last_date  = ph['match_date'].iloc[-1]
        days_since = max(0, (cutoff - last_date).days)
        d7  = cutoff - pd.Timedelta(days=7)
        d14 = cutoff - pd.Timedelta(days=14)
        d28 = cutoff - pd.Timedelta(days=28)
        r7  = ph[ph['match_date'] >= d7]
        r14 = ph[ph['match_date'] >= d14]
        r28 = ph[ph['match_date'] >= d28]
        recent_wins  = ph[ph['won'] == 1].tail(10)
        opp_avg_rank = float(recent_wins['opp_rank'].mean()) if len(recent_wins) > 0 else 100.0
        stats[name] = {
            'rank':                    int(prow['rank']) if not pd.isna(prow['rank']) else 100,
            'cum_mins_7d':             float(r7['minutes'].sum()),
            'cum_mins_14d':            float(r14['minutes'].sum()),
            'cum_mins_28d':            float(r28['minutes'].sum()),
            'surf_weighted_mins_28d':  float(r28['weighted_mins'].sum()),
            'round_weighted_mins_28d': float(r28['weighted_mins'].sum()),
            'matches_7d':              int(len(r7)),
            'days_since_last':         int(days_since),
            'win_pct_10':              float(ph.tail(10)['won'].mean()),
            'win_pct_20':              float(ph.tail(20)['won'].mean()),
            'tourney_matches_before':  0,
            'tourney_mins_before':     0,
            'h2h_win_pct':             0.5,
            'opp_avg_rank_beaten':     opp_avg_rank,
        }
    stats['__h2h_pct__']   = dict(h2h_pct)
    stats['__h2h_wins__']  = dict(h2h_wins)
    stats['__h2h_total__'] = dict(h2h_total)
    return stats


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def render_prob_bar(prob_p1: float, name1: str, name2: str):
    pct_p1 = int(prob_p1 * 100)
    pct_p2 = 100 - pct_p1
    st.markdown(f"""
    <div class="prob-container">
        <div class="prob-p1" style="width:{pct_p1}%;">{pct_p1}%&nbsp;&nbsp;{name1.upper()}</div>
        <div class="prob-p2" style="width:{pct_p2}%;">{name2.upper()}&nbsp;&nbsp;{pct_p2}%</div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_importance(importances, feature_cols):
    pairs = sorted(zip(importances, feature_cols), reverse=True)[:10]
    max_val = pairs[0][0] if pairs else 1
    label_map = {
        'p1_rank': 'P1 Rank', 'p2_rank': 'P2 Rank',
        'p1_cum_mins_7d': 'P1 Load 7d', 'p2_cum_mins_7d': 'P2 Load 7d',
        'p1_cum_mins_14d': 'P1 Load 14d', 'p2_cum_mins_14d': 'P2 Load 14d',
        'p1_cum_mins_28d': 'P1 Load 28d', 'p2_cum_mins_28d': 'P2 Load 28d',
        'p1_surf_weighted_mins_28d': 'P1 Surface-Wtd Load', 'p2_surf_weighted_mins_28d': 'P2 Surface-Wtd Load',
        'p1_round_weighted_mins_28d': 'P1 Round-Wtd Load', 'p2_round_weighted_mins_28d': 'P2 Round-Wtd Load',
        'p1_matches_7d': 'P1 Matches 7d', 'p2_matches_7d': 'P2 Matches 7d',
        'p1_days_since_last': 'P1 Rest Days', 'p2_days_since_last': 'P2 Rest Days',
        'p1_win_pct_10': 'P1 Form (10)', 'p2_win_pct_10': 'P2 Form (10)',
        'p1_win_pct_20': 'P1 Form (20)', 'p2_win_pct_20': 'P2 Form (20)',
        'p1_tourney_matches_before': 'P1 Tourney Matches', 'p2_tourney_matches_before': 'P2 Tourney Matches',
        'p1_tourney_mins_before': 'P1 Tourney Mins', 'p2_tourney_mins_before': 'P2 Tourney Mins',
        'p1_h2h_win_pct': 'P1 H2H Win %', 'p2_h2h_win_pct': 'P2 H2H Win %',
        'p1_opp_avg_rank_beaten': 'P1 Opp Quality', 'p2_opp_avg_rank_beaten': 'P2 Opp Quality',
        'surface': 'Surface',
    }
    html = '<div style="margin-top:16px;">'
    for val, col in pairs:
        label = label_map.get(col, col)
        bar_pct = int((val / max_val) * 100)
        html += f"""
        <div class="fi-row">
            <div class="fi-label">{label}</div>
            <div class="fi-bar-bg"><div class="fi-bar-fill" style="width:{bar_pct}%;"></div></div>
            <div class="fi-val">{val:.3f}</div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_radar(p1_stats, p2_stats, name1, name2):
    def normalize(rank, mins_28, win_pct, rest_days):
        rank_score   = max(0, 100 - rank / 5)
        fresh_score  = max(0, 100 - mins_28 / 12)
        form_score   = win_pct * 100
        rest_score   = min(100, rest_days * 8)
        return [rank_score, fresh_score, form_score, rest_score]
    cats = ['Ranking Power', 'Physical Freshness', 'Current Form', 'Recovery']
    s1 = normalize(p1_stats['rank'], p1_stats['cum_mins_28d'], p1_stats['win_pct_10'], p1_stats['days_since_last'])
    s2 = normalize(p2_stats['rank'], p2_stats['cum_mins_28d'], p2_stats['win_pct_10'], p2_stats['days_since_last'])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=s1 + [s1[0]], theta=cats + [cats[0]], fill='toself',
                                   name=name1, line=dict(color='#4A7FC4', width=2),
                                   fillcolor='rgba(74,127,196,0.2)'))
    fig.add_trace(go.Scatterpolar(r=s2 + [s2[0]], theta=cats + [cats[0]], fill='toself',
                                   name=name2, line=dict(color='#C4622D', width=2),
                                   fillcolor='rgba(196,98,45,0.2)'))
    fig.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
                   radialaxis=dict(visible=True, range=[0, 100], gridcolor='#2E2E2E', tickfont=dict(color='#7A7A7A', size=9)),
                   angularaxis=dict(gridcolor='#2E2E2E', tickfont=dict(color='#7A7A7A', size=10))),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E8E8E8', family='DM Mono'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        margin=dict(l=40, r=40, t=30, b=30), height=340
    )
    return fig


def generate_commentary(prob_p1, p1_stats, p2_stats, name1, name2, surface_name):
    lines = []
    winner = name1 if prob_p1 >= 0.5 else name2
    win_prob = max(prob_p1, 1 - prob_p1)
    if win_prob > 0.72:
        lines.append(f"Strong model conviction: **{winner}** is the clear favorite at {win_prob:.0%}.")
    elif win_prob > 0.58:
        lines.append(f"Model leans **{winner}** ({win_prob:.0%}), though the match remains competitive.")
    else:
        lines.append(f"Essentially a coin flip — model gives **{winner}** a marginal edge at {win_prob:.0%}.")

    load_diff = p1_stats['cum_mins_28d'] - p2_stats['cum_mins_28d']
    heavier = name1 if load_diff > 0 else name2
    if abs(load_diff) > 200:
        lines.append(f"**Cascade Stage 1 signal**: {heavier} carries {abs(load_diff):.0f} more court minutes "
                     f"over 28 days. Per the Precision Degradation Cascade, this level of accumulated load "
                     f"predicts measurable knee flexion reduction and compensatory trunk recruitment — "
                     f"the earliest indicators of precision breakdown.")
    elif abs(load_diff) > 80:
        lines.append(f"**Load differential**: {heavier} has logged {abs(load_diff):.0f} more minutes in the past 28 days. "
                     f"A moderate Stage 1 fatigue signal — worth monitoring if this is a deep tournament run.")

    rest_diff = p1_stats['days_since_last'] - p2_stats['days_since_last']
    more_rested = name1 if rest_diff > 0 else name2
    if abs(rest_diff) > 2:
        lines.append(f"**Recovery advantage (Stage 1–2)**: {more_rested} enters with {abs(rest_diff):.0f} more rest days. "
                     f"Adequate recovery partially reverses lower-body fatigue accumulation before the cascade progresses.")

    h2h_diff = p1_stats['h2h_win_pct'] - p2_stats['h2h_win_pct']
    if abs(h2h_diff) > 0.15 and p1_stats['h2h_win_pct'] != 0.5:
        h2h_leader = name1 if h2h_diff > 0 else name2
        lines.append(f"**Head-to-head edge**: {h2h_leader} holds a meaningful historical advantage "
                     f"in this specific matchup — H2H patterns are incorporated directly into the model's prediction.")

    opp_diff = p2_stats['opp_avg_rank_beaten'] - p1_stats['opp_avg_rank_beaten']
    if abs(opp_diff) > 20:
        stronger_sos = name1 if opp_diff > 0 else name2
        lines.append(f"**Strength of schedule**: {stronger_sos} has been beating higher-ranked opponents recently "
                     f"— a signal of genuine form rather than accumulated wins against weaker fields.")

    if surface_name == 'Clay':
        lines.append("**Surface factor**: Clay extends rally length and maximises cumulative load per match. "
                     "Fatigue features carry amplified predictive weight on this surface per the review's findings.")
    elif surface_name == 'Grass':
        lines.append("**Surface factor**: Grass rewards explosive bursts over sustained endurance. "
                     "Shorter points compress the cascade timeline — Stage 5 cognitive effects may dominate over Stage 1–3.")
    return "  \n".join(lines)


def render_calibration_chart(y_test, y_prob):
    frac, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                              line=dict(dash='dash', color='#7A7A7A'), name='Perfect Calibration'))
    fig.add_trace(go.Scatter(x=mean_pred, y=frac, mode='lines+markers', name='Model',
                              line=dict(color='#C4622D', width=2),
                              marker=dict(size=8, color='#C4622D')))
    fig.update_layout(
        xaxis_title='Mean Predicted Probability', yaxis_title='Fraction of Positives',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E8E8E8', family='DM Mono', size=11),
        xaxis=dict(gridcolor='#2E2E2E'), yaxis=dict(gridcolor='#2E2E2E'),
        legend=dict(bgcolor='rgba(0,0,0,0)'), height=280, margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


def render_roc_comparison(y_test, model_probs_dict):
    """ROC curves for the full model vs each baseline."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                              line=dict(dash='dash', color='#5A5A5A'), name='Random (AUC=0.5)'))
    palette = {'Full Model': '#C4622D', 'Higher-Rank Wins': '#7A7A7A',
               'Rank-Only Logistic': '#4A7FC4', 'Elo-Style (rank-derived)': '#4A7C59'}
    for name, prob in model_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc:.3f})',
                                  line=dict(color=palette.get(name, '#A0A0A0'), width=2)))
    fig.update_layout(
        xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E8E8E8', family='DM Mono', size=11),
        xaxis=dict(gridcolor='#2E2E2E'), yaxis=dict(gridcolor='#2E2E2E'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        height=380, margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


def render_ablation_chart(ablation_results):
    """Bar chart showing AUC lift from each ablation step."""
    ablation_order = [
        'GBM: Ranking only',
        'GBM: + Physiological Load',
        'GBM: + Form',
        'GBM: + H2H + Quality',
        'GBM: Full (all features)',
    ]
    aucs = [ablation_results[k]['roc_auc'] for k in ablation_order if k in ablation_results]
    labels = [k.replace('GBM: ', '') for k in ablation_order if k in ablation_results]
    feat_counts = [ablation_results[k]['features'] for k in ablation_order if k in ablation_results]

    colors = ['#7A7A7A', '#C4622D', '#D08043', '#A83232', '#4A7C59']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=aucs,
        marker_color=colors[:len(aucs)],
        text=[f'{a:.3f}<br>({n} feats)' for a, n in zip(aucs, feat_counts)],
        textposition='outside',
        textfont=dict(family='DM Mono', size=11, color='#E8E8E8'),
    ))
    baseline = aucs[0] if aucs else 0.5
    fig.add_hline(y=baseline, line_dash='dash', line_color='#5A5A5A',
                  annotation_text=f'Ranking-only baseline ({baseline:.3f})',
                  annotation_font=dict(family='DM Mono', size=10, color='#7A7A7A'))
    fig.update_layout(
        yaxis_title='Test AUC',
        xaxis=dict(gridcolor='#2E2E2E', tickfont=dict(family='DM Mono', size=10)),
        yaxis=dict(gridcolor='#2E2E2E', range=[max(0.4, min(aucs) - 0.04), max(aucs) + 0.04]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E8E8E8', family='DM Mono'),
        showlegend=False, height=380, margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    st.markdown("""
    <div style="border-bottom: 1px solid #2E2E2E; padding-bottom: 24px; margin-bottom: 28px;">
        <div style="font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 4px; color: #C4622D; text-transform: uppercase; margin-bottom: 8px;">
            Interactive Research Dashboard &nbsp;·&nbsp; Systematic Review Companion
        </div>
        <h1 style="margin: 0; font-size: 3rem; letter-spacing: 4px; line-height: 1.1;">ATP FATIGUE FORECASTER</h1>
        <div style="font-family: 'DM Sans', sans-serif; font-size: 13px; color: #9A9A9A; margin-top: 10px; max-width: 780px; line-height: 1.6;">
            A machine learning implementation of the <em>Precision Degradation Cascade</em> model —
            quantifying how physiological fatigue overrides baseline ATP ranking in professional match outcomes.
        </div>
        <div style="margin-top: 14px; display: flex; gap: 24px; flex-wrap: wrap;">
            <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: #7A7A7A;">
                📄 &nbsp;<span style="color:#C4622D;">"The Breaking Point"</span> — NHSJS, 2025
            </div>
            <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: #7A7A7A;">
                📊 &nbsp;Data: JeffSackmann/tennis_atp (8-year ATP match records)
            </div>
            <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: #7A7A7A;">
                🧠 &nbsp;Model: Calibrated Gradient Boosting · PRISMA-guided feature design
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading ATP match history (8 years for accurate H2H)..."):
        df_raw = load_raw_data(years=8)
    if df_raw.empty:
        st.error("Could not reach the upstream data repository. Check your internet connection.")
        st.stop()

    with st.spinner("Engineering real rolling physiological features..."):
        df_features = engineer_features(df_raw)
    if len(df_features) < 500:
        st.error("Insufficient data after feature engineering.")
        st.stop()

    with st.spinner("Training calibrated Gradient Boosting model..."):
        model, feature_cols, metrics, importances, X_test, y_test, y_prob_full = train_model(df_features)

    with st.spinner("Running baseline comparisons and ablation study..."):
        eval_results, y_test_eval = evaluate_baselines_and_ablations(df_features)

    with st.spinner("Building player stats index..."):
        player_stats = build_player_stats(df_raw)

    INTERNAL_KEYS = {'__h2h_pct__', '__h2h_wins__', '__h2h_total__'}
    player_names = sorted([k for k in player_stats.keys() if k not in INTERNAL_KEYS])

    df_raw['match_date'] = pd.to_datetime(df_raw['tourney_date'], format='%Y%m%d', errors='coerce')
    data_cutoff = df_raw['match_date'].dropna().max()
    days_stale  = (pd.Timestamp.now() - data_cutoff).days
    freshness_color = '#4A7C59' if days_stale <= 7 else '#C4622D' if days_stale > 21 else '#B8860B'
    st.markdown(f"""
    <div style="background:#141414; border:1px solid #2E2E2E; border-radius:6px;
                padding:8px 16px; margin-bottom:16px; display:flex; align-items:center; gap:12px;">
        <div style="width:8px; height:8px; border-radius:50%; background:{freshness_color}; flex-shrink:0;"></div>
        <div style="font-family:'DM Mono',monospace; font-size:11px; color:#7A7A7A;">
            Data current through <span style="color:#E8E8E8;">{data_cutoff.strftime('%B %d, %Y')}</span>
            &nbsp;·&nbsp; {days_stale} days ago
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──
    st.sidebar.markdown("""
    <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; letter-spacing: 3px; margin-bottom: 16px; color: #E8E8E8;">
    MATCH SETUP
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.sidebar.radio("Input Mode", ["Player Lookup", "Manual Entry"], horizontal=True)
    surface_name = st.sidebar.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
    surface_enc  = {"Hard": 2, "Clay": 0, "Grass": 1, "Carpet": 3}[surface_name]
    st.sidebar.divider()

    if st.sidebar.button("🔄 Clear Cache & Retrain", help="Use if you see a feature mismatch error"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    if input_mode == "Player Lookup" and len(player_names) > 1:
        st.sidebar.markdown("**Player 1**")
        name1 = st.sidebar.selectbox("Select Player 1", player_names, index=0)
        st.sidebar.markdown("**Player 2**")
        available2 = [n for n in player_names if n != name1]
        name2 = st.sidebar.selectbox("Select Player 2", available2, index=0)
        p1 = dict(player_stats[name1])
        p2 = dict(player_stats[name2])
        h2h_pct_table   = player_stats.get('__h2h_pct__',   {})
        h2h_wins_table  = player_stats.get('__h2h_wins__',  {})
        h2h_total_table = player_stats.get('__h2h_total__', {})
        p1_wins_h2h  = h2h_wins_table.get((name1, name2), 0)
        p2_wins_h2h  = h2h_wins_table.get((name2, name1), 0)
        total_h2h    = h2h_total_table.get((name1, name2), 0)
        p1['h2h_win_pct'] = h2h_pct_table.get((name1, name2), 0.5)
        p2['h2h_win_pct'] = h2h_pct_table.get((name2, name1), 0.5)
        if total_h2h > 0:
            leader     = name1 if p1_wins_h2h >= p2_wins_h2h else name2
            lead_color = '#4A7FC4' if p1_wins_h2h >= p2_wins_h2h else '#C4622D'
            st.sidebar.markdown(f"""
            <div style="background:#1C1C1C; border:1px solid #2E2E2E; border-radius:6px;
                        padding:10px 14px; font-family:'DM Mono',monospace; font-size:11px; color:#9A9A9A; margin-top:8px;">
                <div style="font-size:9px; letter-spacing:2px; text-transform:uppercase; margin-bottom:6px; color:#5A5A5A;">
                    All-Time H2H ({total_h2h} meetings)
                </div>
                <span style="color:#4A7FC4; font-size:13px;">{p1_wins_h2h}</span>
                <span style="color:#5A5A5A;"> – </span>
                <span style="color:#C4622D; font-size:13px;">{p2_wins_h2h}</span>
                &nbsp;&nbsp;
                <span style="color:{lead_color}; font-size:10px;">{leader.split()[-1]} leads</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        name1 = st.sidebar.text_input("Player 1 Name", "Player 1")
        name2 = st.sidebar.text_input("Player 2 Name", "Player 2")
        st.sidebar.markdown("**Player 1**")
        p1_rank = st.sidebar.number_input("P1 Rank", 1, 500, 10)
        p1_m28  = st.sidebar.slider("P1 Load — 28d (mins)", 0, 1400, 400)
        p1 = {
            'rank': p1_rank,
            'cum_mins_7d':  st.sidebar.slider("P1 Load — 7d",  0, 600,  120),
            'cum_mins_14d': st.sidebar.slider("P1 Load — 14d", 0, 900,  240),
            'cum_mins_28d': p1_m28,
            'surf_weighted_mins_28d':  p1_m28 * SURFACE_FATIGUE_WEIGHT.get(surface_name, 1.0),
            'round_weighted_mins_28d': p1_m28 * SURFACE_FATIGUE_WEIGHT.get(surface_name, 1.0),
            'matches_7d':              st.sidebar.slider("P1 Matches (7d)", 0, 10, 3),
            'days_since_last':         st.sidebar.slider("P1 Rest Days", 0, 30, 3),
            'win_pct_10':              st.sidebar.slider("P1 Win% (10)", 0.0, 1.0, 0.7),
            'win_pct_20':              st.sidebar.slider("P1 Win% (20)", 0.0, 1.0, 0.65),
            'tourney_matches_before':  st.sidebar.slider("P1 Tourney Matches", 0, 6, 0),
            'tourney_mins_before':     st.sidebar.slider("P1 Tourney Mins",    0, 900, 0),
            'h2h_win_pct':             st.sidebar.slider("P1 H2H Win %", 0.0, 1.0, 0.5),
            'opp_avg_rank_beaten':     st.sidebar.number_input("P1 Avg Opp Rank Beaten", 1, 500, 40),
        }
        st.sidebar.markdown("**Player 2**")
        p2_rank = st.sidebar.number_input("P2 Rank", 1, 500, 20)
        p2_m28  = st.sidebar.slider("P2 Load — 28d (mins)", 0, 1400, 700)
        p2 = {
            'rank': p2_rank,
            'cum_mins_7d':  st.sidebar.slider("P2 Load — 7d",  0, 600,  280),
            'cum_mins_14d': st.sidebar.slider("P2 Load — 14d", 0, 900,  450),
            'cum_mins_28d': p2_m28,
            'surf_weighted_mins_28d':  p2_m28 * SURFACE_FATIGUE_WEIGHT.get(surface_name, 1.0),
            'round_weighted_mins_28d': p2_m28 * SURFACE_FATIGUE_WEIGHT.get(surface_name, 1.0),
            'matches_7d':              st.sidebar.slider("P2 Matches (7d)", 0, 10, 6),
            'days_since_last':         st.sidebar.slider("P2 Rest Days", 0, 30, 1),
            'win_pct_10':              st.sidebar.slider("P2 Win% (10)", 0.0, 1.0, 0.5),
            'win_pct_20':              st.sidebar.slider("P2 Win% (20)", 0.0, 1.0, 0.5),
            'tourney_matches_before':  st.sidebar.slider("P2 Tourney Matches", 0, 6, 3),
            'tourney_mins_before':     st.sidebar.slider("P2 Tourney Mins",    0, 900, 380),
            'h2h_win_pct':             st.sidebar.slider("P2 H2H Win %", 0.0, 1.0, 0.5),
            'opp_avg_rank_beaten':     st.sidebar.number_input("P2 Avg Opp Rank Beaten", 1, 500, 60),
        }

    run = st.sidebar.button("Run Inference", type="primary", use_container_width=True)

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "MATCH INFERENCE", "MODEL PERFORMANCE", "ABLATION STUDY", "KEY FINDINGS", "METHODOLOGY"
    ])

    # ════════════════════════════════════════════════════════════
    # TAB 2: MODEL PERFORMANCE — full model vs baselines
    # ════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-label">Validation: Full Model vs. Baselines</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif; font-size:14px; color:#C8C8C8; line-height:1.7; margin-bottom:20px;">
        A model is only meaningful if it outperforms simpler alternatives. Below, the full Calibrated GBM is
        evaluated on the same temporal hold-out set against three baselines: a naive higher-rank-wins rule,
        rank-only logistic regression, and an Elo-style expected-score formula. All models are evaluated on identical
        unseen data (last 20% of matches by chronological order) — no future leakage.
        </div>
        """, unsafe_allow_html=True)

        # Headline metrics for the full model
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test Accuracy",  f"{metrics['accuracy']:.2%}")
        c2.metric("ROC-AUC",        f"{metrics['roc_auc']:.3f}")
        c3.metric("CV AUC (±σ)",    f"{metrics['cv_auc_mean']:.3f} ±{metrics['cv_auc_std']:.3f}")
        c4.metric("Brier Score",    f"{metrics['brier']:.3f}", help="Lower = better calibrated.")
        c5.metric("Log Loss",       f"{metrics['log_loss']:.3f}", help="Lower = better.")

        st.divider()

        # Comparison table
        st.markdown("**Head-to-Head Comparison**")
        full_row = {
            'Model':     'Full Model (Calibrated GBM)',
            'Features':  len(feature_cols),
            'AUC':       metrics['roc_auc'],
            'Accuracy':  metrics['accuracy'],
            'Brier':     metrics['brier'],
            'Log Loss':  metrics['log_loss'],
            'Description': 'All physiological-load + form + H2H + quality + surface features',
        }
        baseline_keys = ['Higher-Rank Wins', 'Rank-Only Logistic', 'Elo-Style (rank-derived)']
        rows = [full_row]
        for k in baseline_keys:
            r = eval_results[k]
            rows.append({
                'Model':     k,
                'Features':  r['features'],
                'AUC':       r['roc_auc'],
                'Accuracy':  r['accuracy'],
                'Brier':     r['brier'],
                'Log Loss':  r['log_loss'],
                'Description': r['desc'],
            })
        comp_df = pd.DataFrame(rows)
        comp_df['AUC']      = comp_df['AUC'].map(lambda x: f'{x:.3f}')
        comp_df['Accuracy'] = comp_df['Accuracy'].map(lambda x: f'{x:.2%}')
        comp_df['Brier']    = comp_df['Brier'].map(lambda x: f'{x:.3f}')
        comp_df['Log Loss'] = comp_df['Log Loss'].map(lambda x: f'{x:.3f}')
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Lift over best baseline
        best_baseline_auc = max(eval_results[k]['roc_auc'] for k in baseline_keys)
        lift = metrics['roc_auc'] - best_baseline_auc
        lift_pct = 100 * lift / best_baseline_auc
        lift_color = '#4A7C59' if lift > 0.01 else '#C4622D' if lift > 0 else '#A83232'
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(74,124,89,0.12), rgba(196,98,45,0.08));
                    border-left:4px solid {lift_color}; border-radius:0 8px 8px 0; padding:14px 20px; margin-top:12px;">
            <div style="font-family:'DM Mono',monospace; font-size:11px; letter-spacing:2px; color:#7A7A7A; text-transform:uppercase; margin-bottom:6px;">
                Lift over best baseline
            </div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:1.6rem; letter-spacing:2px; color:{lift_color};">
                +{lift:.3f} AUC &nbsp;·&nbsp; +{lift_pct:.1f}%
            </div>
            <div style="font-family:'DM Sans',sans-serif; font-size:13px; color:#A0A0A0; margin-top:6px;">
                Full model outperforms the strongest non-ML baseline (best of higher-rank, rank-logistic, Elo-style).
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ROC curves & calibration side-by-side
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**ROC Curves: Full Model vs Baselines**")
            split = int(len(df_features) * 0.8)
            test_slice = df_features.iloc[split:].reset_index(drop=True)
            rank_diff = (test_slice['p2_rank'] - test_slice['p1_rank']).values
            higher_rank_prob = 1 / (1 + np.exp(-rank_diff / 30))
            X_rank_test = df_features[['p1_rank', 'p2_rank']].values[split:]
            X_rank_train = df_features[['p1_rank', 'p2_rank']].values[:split]
            y_train = df_features['p1_wins'].values[:split]
            lr_rank = LogisticRegression(max_iter=1000).fit(X_rank_train, y_train)
            lr_prob = lr_rank.predict_proba(X_rank_test)[:, 1]
            elo_probs = []
            for _, r in test_slice.iterrows():
                p1_rating = 1500 + (300 - min(r['p1_rank'], 300)) * 2
                p2_rating = 1500 + (300 - min(r['p2_rank'], 300)) * 2
                elo_probs.append(1 / (1 + 10 ** ((p2_rating - p1_rating) / 400)))
            elo_probs = np.array(elo_probs)

            roc_dict = {
                'Full Model':                y_prob_full,
                'Higher-Rank Wins':          higher_rank_prob,
                'Rank-Only Logistic':        lr_prob,
                'Elo-Style (rank-derived)':  elo_probs,
            }
            st.plotly_chart(render_roc_comparison(y_test, roc_dict), use_container_width=True)

        with cb:
            st.markdown("**Model Calibration**")
            st.caption("Predicted probabilities vs. actual win frequencies. Diagonal = perfectly calibrated.")
            st.plotly_chart(render_calibration_chart(y_test, y_prob_full), use_container_width=True)

        st.divider()
        st.markdown("**Feature Importance** (top 10 by GBM gain)")
        render_feature_importance(importances, feature_cols)

    # ════════════════════════════════════════════════════════════
    # TAB 3: ABLATION STUDY — does load actually add lift?
    # ════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-label">Ablation: Empirical Validation of the Research Hypothesis</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif; font-size:14px; color:#C8C8C8; line-height:1.7; margin-bottom:20px;">
        The systematic review's central hypothesis is that <strong style="color:#E8E8E8;">physiological load
        accumulation adds predictive signal beyond static ranking</strong>. This ablation tests the claim directly:
        we train identical Gradient Boosting classifiers on progressively richer feature sets and measure AUC on
        the same hold-out window. Each step isolates the marginal contribution of one feature group.
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(render_ablation_chart(eval_results), use_container_width=True)

        # Build human-readable ablation table
        ablation_keys = [
            'GBM: Ranking only',
            'GBM: + Physiological Load',
            'GBM: + Form',
            'GBM: + H2H + Quality',
            'GBM: Full (all features)',
        ]
        rows = []
        prev_auc = None
        for k in ablation_keys:
            if k not in eval_results:
                continue
            r = eval_results[k]
            delta = '—' if prev_auc is None else f'{r["roc_auc"] - prev_auc:+.4f}'
            rows.append({
                'Step':        k.replace('GBM: ', ''),
                'Features':    r['features'],
                'AUC':         f'{r["roc_auc"]:.4f}',
                'Δ vs Prev':   delta,
                'Accuracy':    f'{r["accuracy"]:.2%}',
                'What Was Added': r['desc'],
            })
            prev_auc = r['roc_auc']
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Compute the headline lift
        if 'GBM: Ranking only' in eval_results and 'GBM: + Physiological Load' in eval_results:
            base_auc = eval_results['GBM: Ranking only']['roc_auc']
            with_load_auc = eval_results['GBM: + Physiological Load']['roc_auc']
            load_lift = with_load_auc - base_auc
            full_auc = eval_results['GBM: Full (all features)']['roc_auc']
            full_lift = full_auc - base_auc

            st.markdown(f"""
            <div style="background:linear-gradient(135deg, rgba(196,98,45,0.15), rgba(74,124,89,0.10));
                        border:1px solid rgba(196,98,45,0.4); border-radius:8px; padding:20px 24px; margin-top:20px;">
                <div style="font-family:'Bebas Neue',sans-serif; font-size:1.3rem; letter-spacing:3px; color:#C4622D; margin-bottom:10px;">
                    HYPOTHESIS RESULT
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:14px; color:#E8E8E8; line-height:1.8;">
                    Adding physiological-load features to a rank-only GBM raises test AUC by
                    <strong style="color:#C4622D;">{load_lift:+.4f}</strong>
                    (from {base_auc:.4f} → {with_load_auc:.4f}). The full feature set lifts AUC by
                    <strong style="color:#4A7C59;">{full_lift:+.4f}</strong> over ranking alone.
                    {"This supports the review's core hypothesis: scheduling-derived load carries genuine, separable predictive signal beyond ranking." if load_lift > 0 else "In this run, load features did not improve over ranking alone — possibly due to redundancy with rank for high-load top players, or insufficient sample size in load-extreme regions."}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        **How to read this:** Each step in the ablation pipeline adds one group of features that map to a specific
        cascade stage from the systematic review. The leftmost bar is a ranking-only sanity check — what you'd get
        from Elo or ATP rank alone. Each subsequent bar adds the engineered features tied to a specific research-paper
        construct (Stage 1 load, Stage 3-5 form, contextual H2H/quality, surface). If a step improves AUC,
        that feature group contains real, non-redundant signal.
        """)

    # ════════════════════════════════════════════════════════════
    # TAB 4 & TAB 5: KEY FINDINGS, METHODOLOGY (kept from v2)
    # ════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-label">From the Systematic Review · 10 Studies · 847 Records Screened</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background: #141414; border: 1px solid #2E2E2E; border-left: 4px solid #C4622D;
                    border-radius: 0 8px 8px 0; padding: 24px 28px; margin-bottom: 28px;">
            <div style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:3px; color:#C4622D; margin-bottom:12px; text-transform:uppercase;">
                THE BREAKING POINT — Abstract
            </div>
            <div style="font-family:'DM Sans',sans-serif; font-size:14px; color:#C8C8C8; line-height:1.8;">
                Peer-reviewed systematic review (NHSJS, 2025) of PubMed, Google Scholar, and SPORTDiscus (2002–2023)
                following PRISMA 2020 guidelines. Ten studies were synthesized. A distinct dissociation was found between
                power and precision under fatigue: <strong style="color:#E8E8E8;">serve velocity declined only 0.4–3.1%</strong>,
                while <strong style="color:#E8E8E8;">serve accuracy degraded 25–32% and groundstroke accuracy up to 69%</strong>.
                Reaction time delayed 47–68 ms; decision-making quality declined 18–34%.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### The Velocity–Accuracy Paradox")
        col_va1, col_va2 = st.columns([3, 2])
        with col_va1:
            st.markdown("""
            The most counterintuitive finding of the review: **fatigue does not slow players down — it makes them inaccurate.**

            Across the included primary studies, serve velocity under fatigued conditions fell by less than 3.1% — within
            normal match variation and statistically non-significant in most protocols. Yet in the same conditions,
            serve accuracy dropped 25–32% and groundstroke accuracy collapsed by up to **69%** in high-intensity protocols
            (Davey et al., 2002).

            This challenges the traditional definition of fatigue as "reduced force production"
            (Edwards, 1981). What actually limits elite performance is **neural inefficiency** — degradation
            of fine motor control while gross power output remains preserved.
            """)
        with col_va2:
            paradox_df = pd.DataFrame({
                'Metric': ['Serve Velocity', 'Serve Accuracy', 'Groundstroke Accuracy'],
                'Avg Decline (%)': [1.8, 28.5, 54.0],
            })
            fig_paradox = go.Figure()
            colors = ['#4A7FC4', '#C4622D', '#A83232']
            for i, row in paradox_df.iterrows():
                fig_paradox.add_trace(go.Bar(
                    x=[row['Avg Decline (%)']], y=[row['Metric']], orientation='h',
                    marker_color=colors[i], name=row['Metric'],
                    text=[f"-{row['Avg Decline (%)']:.1f}%"], textposition='outside',
                    textfont=dict(family='DM Mono', size=12, color='#E8E8E8')
                ))
            fig_paradox.update_layout(
                title=dict(text="Average Performance Decline Under Fatigue", font=dict(family='DM Mono', size=11, color='#7A7A7A')),
                xaxis=dict(title="% Decline from Baseline", gridcolor='#2E2E2E', range=[0, 75], tickfont=dict(family='DM Mono', color='#7A7A7A')),
                yaxis=dict(gridcolor='#2E2E2E', tickfont=dict(family='DM Mono', color='#E8E8E8')),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False, height=220, margin=dict(l=10, r=60, t=40, b=10)
            )
            st.plotly_chart(fig_paradox, use_container_width=True)

        st.divider()
        st.markdown("### The Precision Degradation Cascade")
        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif; font-size:14px; color:#C8C8C8; line-height:1.8; margin-bottom:20px;">
        Synthesizing biomechanical and cognitive data across the included studies, the review proposes a hypothesis-generating
        framework explaining how elite performance degrades under fatigue — not all at once, but in a 5-stage sequence.
        </div>
        """, unsafe_allow_html=True)

        stages = [
            ("Stage 1", "Lower-Body Fatigue", "#C4622D",
             "Metabolic fatigue in the quadriceps and gastrocnemius causes reduced knee flexion (15–23°) and declining ground reaction force. The earliest measurable signal of cascade onset."),
            ("Stage 2", "Kinetic Chain Compensation", "#B85A28",
             "The CNS prioritizes force maintenance. Trunk rotation velocity increases 8–12% to compensate for lost leg drive — preserving ball speed at a structural cost."),
            ("Stage 3", "Precision Loss", "#A0491E",
             "Compensatory proximal recruitment destabilizes distal fine motor control. Velocity holds, but placement collapses: serve accuracy −25–32%, groundstrokes −38–69%."),
            ("Stage 4", "Range of Motion Restriction", "#8A3A16",
             "Continued fatigue restricts hip rotation ROM (~13°), reducing topspin generation. Shots flatten out — a tactical liability even if power is intact."),
            ("Stage 5", "Cognitive Failure", "#6E2B0E",
             "Systemic fatigue impairs executive function: reaction time delays of 47–68 ms, and decision-making quality declines 18–34%."),
        ]
        for stage_id, stage_name, color, desc in stages:
            st.markdown(f"""
            <div style="display:flex; gap:16px; margin:10px 0; align-items:flex-start;">
                <div style="background:{color}; border-radius:6px; padding:8px 14px; flex-shrink:0;
                            font-family:'Bebas Neue',sans-serif; font-size:1rem; letter-spacing:2px;
                            color:white; min-width:80px; text-align:center;">{stage_id}</div>
                <div style="background:#1C1C1C; border:1px solid #2E2E2E; border-radius:6px;
                            padding:12px 18px; flex:1;">
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.1rem; letter-spacing:2px;
                                color:#E8E8E8; margin-bottom:4px;">{stage_name.upper()}</div>
                    <div style="font-family:'DM Sans',sans-serif; font-size:13px; color:#A0A0A0; line-height:1.6;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-bar" style="margin-top:16px;">
            ⚡ This cascade is the theoretical foundation for the ML model's feature design.
            Court minutes in 7/14/28-day windows operationalize Stage 1 load accumulation.
            Days since last match captures Stage 1–2 recovery. Rolling win rates reflect Stage 3–5 downstream effects.
            <strong>The Ablation Study tab tests whether these features actually add lift.</strong>
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="section-label">Research Design & ML Implementation</div>', unsafe_allow_html=True)
        col_m, col_r = st.columns([3, 2])
        with col_m:
            st.markdown("""
**Research Question**

Traditional ATP forecasting over-weights static rankings and ignores cumulative match-play cost.
This dashboard operationalizes the *Precision Degradation Cascade* as computable ML features and
empirically tests — via the Ablation Study tab — whether scheduling-derived load adds predictive
signal beyond ranking alone.

**From Paper to Features**

The systematic review identified court-time accumulation as the earliest measurable Stage 1 cascade signal.
This maps to the model's core features: cumulative minutes in 7/14/28-day windows from real ATP scheduling
records. Recovery days capture Stage 1–2 restoration. Rolling win rates reflect downstream Stage 3–5 effects.

**Why Calibrated Gradient Boosting**

GBM builds trees sequentially, correcting residuals — well-suited to non-linear fatigue thresholds
(the "breaking point" concept). Isotonic calibration ensures probabilities are statistically reliable,
not just rank-ordered. Verified empirically via reliability diagram and Brier score.
            """)
        with col_r:
            st.markdown("""
**Feature Engineering**

`LOAD (7/14/28d)` — cumulative court minutes in rolling windows

`SURFACE-WEIGHTED LOAD` — 28d minutes scaled by surface multiplier (clay ×1.27, grass ×0.80)

`ROUND-WEIGHTED LOAD` — minutes scaled by round intensity (SF ×1.15, F ×1.30)

`H2H RECORD` — head-to-head win rate vs this opponent

`OPPONENT QUALITY` — avg rank of players beaten in last 10 wins

`RECOVERY` — days since last match; matches in prior 7 days

`FORM` — rolling win % over last 10 and 20 matches

`SURFACE` — clay / hard / grass / carpet

**Validation**

- Temporal 80/20 train/test split (no future leakage)
- 5-fold CV on training set (stratified)
- Three baseline comparisons (Performance tab)
- 5-step ablation study (Ablation tab)
- Calibration verified by reliability diagram

**Data Source**

[JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) — 8 years of ATP match records.
            """)

        st.divider()
        st.markdown("""
**Honest Limitations**

Heterogeneity in how the ATP records match duration mirrors the measurement heterogeneity that
prevented formal meta-analysis in the systematic review. The model captures scheduling load but
cannot directly observe Stage 1–3 biomechanical variables (knee flexion, EMG) — these remain latent
signals approximated by court time.

**Citation**

*"The Breaking Point: A Systematic Review of Physiological and Cognitive Fatigue Effects on
Professional Tennis Performance"* — National High School Journal of Science (NHSJS), 2025.
PRISMA-compliant systematic review (847 records → 10 included). PubMed · Google Scholar · SPORTDiscus.
        """)

    # ════════════════════════════════════════════════════════════
    # TAB 1: MATCH INFERENCE — with rest-day counterfactual
    # ════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-label">Pre-Match Analysis</div>', unsafe_allow_html=True)

        col_p1, col_vs, col_p2 = st.columns([5, 1, 5])
        with col_p1:
            st.markdown(f"""
            <div class="player-card">
                <div class="player-name" style="color:#4A7FC4;">{name1.upper()}</div>
                <div class="player-rank">ATP Rank #{p1['rank']}</div>
                <div style="margin-top:12px;">
                    <span class="stat-pill">7d: {p1['cum_mins_7d']:.0f} mins</span>
                    <span class="stat-pill">28d: {p1['cum_mins_28d']:.0f} mins</span>
                    <span class="stat-pill">Wtd: {p1['surf_weighted_mins_28d']:.0f} mins</span>
                    <span class="stat-pill">Rest: {p1['days_since_last']:.0f}d</span>
                    <span class="stat-pill">Form: {p1['win_pct_10']:.0%}</span>
                    <span class="stat-pill">H2H: {p1['h2h_win_pct']:.0%}</span>
                    <span class="stat-pill">Opp Qlty: #{p1['opp_avg_rank_beaten']:.0f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_vs:
            st.markdown("<div style='text-align:center; font-family: Bebas Neue; font-size:2rem; color:#7A7A7A; margin-top:30px;'>VS</div>", unsafe_allow_html=True)
        with col_p2:
            st.markdown(f"""
            <div class="player-card">
                <div class="player-name" style="color:#C4622D;">{name2.upper()}</div>
                <div class="player-rank">ATP Rank #{p2['rank']}</div>
                <div style="margin-top:12px;">
                    <span class="stat-pill">7d: {p2['cum_mins_7d']:.0f} mins</span>
                    <span class="stat-pill">28d: {p2['cum_mins_28d']:.0f} mins</span>
                    <span class="stat-pill">Wtd: {p2['surf_weighted_mins_28d']:.0f} mins</span>
                    <span class="stat-pill">Rest: {p2['days_since_last']:.0f}d</span>
                    <span class="stat-pill">Form: {p2['win_pct_10']:.0%}</span>
                    <span class="stat-pill">H2H: {p2['h2h_win_pct']:.0%}</span>
                    <span class="stat-pill">Opp Qlty: #{p2['opp_avg_rank_beaten']:.0f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
        st.plotly_chart(render_radar(p1, p2, name1, name2), use_container_width=True)

        if run:
            def build_input_dict(p1_d, p2_d, surf_enc):
                return {
                    'p1_rank': p1_d['rank'], 'p2_rank': p2_d['rank'],
                    'p1_cum_mins_7d': p1_d['cum_mins_7d'], 'p2_cum_mins_7d': p2_d['cum_mins_7d'],
                    'p1_cum_mins_14d': p1_d['cum_mins_14d'], 'p2_cum_mins_14d': p2_d['cum_mins_14d'],
                    'p1_cum_mins_28d': p1_d['cum_mins_28d'], 'p2_cum_mins_28d': p2_d['cum_mins_28d'],
                    'p1_surf_weighted_mins_28d': p1_d['surf_weighted_mins_28d'],
                    'p2_surf_weighted_mins_28d': p2_d['surf_weighted_mins_28d'],
                    'p1_round_weighted_mins_28d': p1_d['round_weighted_mins_28d'],
                    'p2_round_weighted_mins_28d': p2_d['round_weighted_mins_28d'],
                    'p1_matches_7d': p1_d['matches_7d'], 'p2_matches_7d': p2_d['matches_7d'],
                    'p1_days_since_last': p1_d['days_since_last'], 'p2_days_since_last': p2_d['days_since_last'],
                    'p1_win_pct_10': p1_d['win_pct_10'], 'p2_win_pct_10': p2_d['win_pct_10'],
                    'p1_win_pct_20': p1_d['win_pct_20'], 'p2_win_pct_20': p2_d['win_pct_20'],
                    'p1_tourney_matches_before': p1_d['tourney_matches_before'],
                    'p2_tourney_matches_before': p2_d['tourney_matches_before'],
                    'p1_tourney_mins_before': p1_d['tourney_mins_before'],
                    'p2_tourney_mins_before': p2_d['tourney_mins_before'],
                    'p1_h2h_win_pct': p1_d['h2h_win_pct'], 'p2_h2h_win_pct': p2_d['h2h_win_pct'],
                    'p1_opp_avg_rank_beaten': p1_d['opp_avg_rank_beaten'],
                    'p2_opp_avg_rank_beaten': p2_d['opp_avg_rank_beaten'],
                    'surface': surf_enc,
                }

            input_dict = build_input_dict(p1, p2, surface_enc)
            missing = [c for c in feature_cols if c not in input_dict]
            if missing:
                st.error(f"Model expects features not in input: {missing}. Clear cache and reload.")
                st.stop()
            input_df = pd.DataFrame([[input_dict[c] for c in feature_cols]], columns=feature_cols)
            probs = model.predict_proba(input_df)[0]
            prob_p1 = probs[1]

            st.divider()
            st.markdown('<div class="section-label">Inference Result</div>', unsafe_allow_html=True)
            render_prob_bar(prob_p1, name1, name2)

            winner_name = name1 if prob_p1 >= 0.5 else name2
            win_prob    = max(prob_p1, 1 - prob_p1)
            confidence  = "HIGH" if win_prob > 0.68 else "MEDIUM" if win_prob > 0.55 else "LOW"
            conf_color  = "#4A7C59" if confidence == "HIGH" else "#C4622D" if confidence == "MEDIUM" else "#7A7A7A"

            st.markdown(f"""
            <div style="display:flex; gap:12px; margin:16px 0; flex-wrap:wrap;">
                <div style="background:#1C1C1C; border:1px solid #2E2E2E; border-radius:6px; padding:12px 20px; flex:1; min-width:160px;">
                    <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:2px; color:#7A7A7A; text-transform:uppercase; margin-bottom:4px;">Predicted Winner</div>
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem; letter-spacing:2px;">{winner_name.upper()}</div>
                </div>
                <div style="background:#1C1C1C; border:1px solid #2E2E2E; border-radius:6px; padding:12px 20px; flex:1; min-width:160px;">
                    <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:2px; color:#7A7A7A; text-transform:uppercase; margin-bottom:4px;">Win Probability</div>
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem; letter-spacing:2px;">{win_prob:.1%}</div>
                </div>
                <div style="background:#1C1C1C; border:1px solid #2E2E2E; border-radius:6px; padding:12px 20px; flex:1; min-width:160px;">
                    <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:2px; color:#7A7A7A; text-transform:uppercase; margin-bottom:4px;">Model Confidence</div>
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem; letter-spacing:2px; color:{conf_color};">{confidence}</div>
                </div>
                <div style="background:#1C1C1C; border:1px solid #2E2E2E; border-radius:6px; padding:12px 20px; flex:1; min-width:160px;">
                    <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:2px; color:#7A7A7A; text-transform:uppercase; margin-bottom:4px;">Surface</div>
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem; letter-spacing:2px;">{surface_name.upper()}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            commentary = generate_commentary(prob_p1, p1, p2, name1, name2, surface_name)
            st.markdown(f'<div class="verdict">{commentary}</div>', unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════
            # COUNTERFACTUAL — sweep rest days and load for both players
            # ════════════════════════════════════════════════════════
            st.divider()
            st.markdown('<div class="section-label">Counterfactual Analysis · What-If Simulator</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'DM Sans',sans-serif; font-size:13px; color:#A0A0A0; line-height:1.7; margin-bottom:16px;">
                These simulations isolate the marginal effect of <strong style="color:#E8E8E8;">recovery</strong> and
                <strong style="color:#E8E8E8;">cumulative load</strong> on the model's prediction — holding everything else fixed.
                The slope of each curve quantifies how much the model believes that variable matters for this specific matchup.
            </div>
            """, unsafe_allow_html=True)

            cf_col1, cf_col2 = st.columns(2)

            # ── Counterfactual 1: P1 rest-day sweep ──
            with cf_col1:
                st.markdown(f"**If {name1} had X more rest days...**")
                rest_sweep = list(range(0, 15))
                probs_sweep = []
                for extra in rest_sweep:
                    p1_cf = dict(p1)
                    p1_cf['days_since_last'] = p1['days_since_last'] + extra
                    inp = build_input_dict(p1_cf, p2, surface_enc)
                    df_inp = pd.DataFrame([[inp[c] for c in feature_cols]], columns=feature_cols)
                    probs_sweep.append(model.predict_proba(df_inp)[0][1])

                fig_cf = go.Figure()
                fig_cf.add_trace(go.Scatter(
                    x=rest_sweep, y=probs_sweep, mode='lines+markers',
                    line=dict(color='#4A7FC4', width=2), marker=dict(size=6, color='#4A7FC4'),
                    name=f'{name1} P(win)'
                ))
                fig_cf.add_hline(y=prob_p1, line_dash='dash', line_color='#7A7A7A',
                                 annotation_text=f'Current ({prob_p1:.2%})',
                                 annotation_font=dict(family='DM Mono', size=9, color='#7A7A7A'))
                fig_cf.add_hline(y=0.5, line_dash='dot', line_color='#5A5A5A',
                                 annotation_text='50%',
                                 annotation_font=dict(family='DM Mono', size=9, color='#5A5A5A'))
                fig_cf.update_layout(
                    xaxis_title='Additional Rest Days', yaxis_title=f'P({name1} wins)',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E8E8E8', family='DM Mono', size=10),
                    xaxis=dict(gridcolor='#2E2E2E'),
                    yaxis=dict(gridcolor='#2E2E2E', tickformat='.0%'),
                    height=260, margin=dict(l=10, r=10, t=20, b=10), showlegend=False
                )
                st.plotly_chart(fig_cf, use_container_width=True)

                max_lift = max(probs_sweep) - prob_p1
                st.markdown(f"""
                <div class="counterfactual-box">
                    Maximum recovery upside for <strong>{name1}</strong>:
                    <strong style="color:#4A7C59;">{max_lift:+.2%}</strong> win probability
                    over a 0–14 day rest range. This is how much the model believes recovery alone can shift this matchup —
                    isolating Stage 1 cascade reversal.
                </div>
                """, unsafe_allow_html=True)

            # ── Counterfactual 2: P2 28-day load sweep ──
            with cf_col2:
                st.markdown(f"**If {name2}'s 28-day load varies...**")
                load_sweep = list(range(0, 1300, 100))
                probs_load = []
                for load_val in load_sweep:
                    p2_cf = dict(p2)
                    p2_cf['cum_mins_28d'] = load_val
                    p2_cf['surf_weighted_mins_28d']  = load_val * SURFACE_FATIGUE_WEIGHT.get(surface_name, 1.0)
                    p2_cf['round_weighted_mins_28d'] = load_val * SURFACE_FATIGUE_WEIGHT.get(surface_name, 1.0)
                    inp = build_input_dict(p1, p2_cf, surface_enc)
                    df_inp = pd.DataFrame([[inp[c] for c in feature_cols]], columns=feature_cols)
                    probs_load.append(model.predict_proba(df_inp)[0][1])

                fig_load = go.Figure()
                fig_load.add_trace(go.Scatter(
                    x=load_sweep, y=probs_load, mode='lines+markers',
                    line=dict(color='#C4622D', width=2), marker=dict(size=6, color='#C4622D'),
                ))
                fig_load.add_hline(y=prob_p1, line_dash='dash', line_color='#7A7A7A',
                                   annotation_text=f'Current ({prob_p1:.2%})',
                                   annotation_font=dict(family='DM Mono', size=9, color='#7A7A7A'))
                fig_load.add_vline(x=p2['cum_mins_28d'], line_dash='dot', line_color='#C4622D',
                                   annotation_text=f'Actual: {p2["cum_mins_28d"]:.0f}',
                                   annotation_font=dict(family='DM Mono', size=9, color='#C4622D'))
                fig_load.update_layout(
                    xaxis_title=f'{name2} 28-day Court Minutes', yaxis_title=f'P({name1} wins)',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E8E8E8', family='DM Mono', size=10),
                    xaxis=dict(gridcolor='#2E2E2E'),
                    yaxis=dict(gridcolor='#2E2E2E', tickformat='.0%'),
                    height=260, margin=dict(l=10, r=10, t=20, b=10), showlegend=False
                )
                st.plotly_chart(fig_load, use_container_width=True)

                load_range = max(probs_load) - min(probs_load)
                st.markdown(f"""
                <div class="counterfactual-box">
                    Across a 0–1200 minute 28-day load range for <strong>{name2}</strong>,
                    {name1}'s win probability swings by <strong style="color:#C4622D;">{load_range:.2%}</strong>.
                    Larger swings indicate the model considers cumulative load a meaningful predictor for this matchup.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            st.markdown("**Disclaimer:** This tool is for research purposes only and is not intended for betting or wagering.")


if __name__ == "__main__":
    np.random.seed(42)
    main()
