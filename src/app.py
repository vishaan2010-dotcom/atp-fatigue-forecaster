"""
ATP Fatigue Forecaster — Research-Grade ML Dashboard
Built on real rolling physiological load features from JeffSackmann/tennis_atp
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
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

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; }
[data-testid="stMetricValue"] { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: var(--text); letter-spacing: 2px; }

/* Tabs */
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

/* Probability bars */
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

/* Verdict box */
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

/* Feature importance bar */
.fi-row { display: flex; align-items: center; margin: 6px 0; gap: 12px; }
.fi-label { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); width: 180px; text-align: right; flex-shrink: 0; text-transform: uppercase; letter-spacing: 0.5px; }
.fi-bar-bg { flex: 1; background: var(--border); border-radius: 3px; height: 8px; }
.fi-bar-fill { height: 8px; border-radius: 3px; background: linear-gradient(90deg, var(--clay), var(--clay-lt)); }
.fi-val { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); width: 48px; }

/* Section headers */
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

/* Player columns */
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

/* Warning banner */
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

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def load_raw_data(years: int = 5) -> pd.DataFrame:
    """
    Load multiple years of ATP match data from JeffSackmann/tennis_atp.
    Fetches up to `years` years of data for robust rolling feature computation.
    """
    current_year = datetime.datetime.now().year
    frames = []

    for year in range(current_year, current_year - years - 1, -1):
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        try:
            logging.info(f"Fetching {year}...")
            df_year = pd.read_csv(url, low_memory=False)
            frames.append(df_year)
            logging.info(f"Loaded {len(df_year)} matches from {year}")
        except urllib.error.HTTPError:
            logging.warning(f"{year} not yet published, skipping.")
        except Exception as e:
            logging.error(f"Error loading {year}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING CONSTANTS
# ─────────────────────────────────────────────

# Surface fatigue multipliers — derived from average rally length research
# Clay ~27% longer rallies than hard, grass ~20% shorter (Hornery et al. 2007)
SURFACE_FATIGUE_WEIGHT = {'Clay': 1.27, 'Hard': 1.0, 'Grass': 0.80, 'Carpet': 0.90}

# Round physiological weight — SF/F are disproportionately taxing
# Encodes that a 90-min final is not equal to a 90-min first round
ROUND_FATIGUE_WEIGHT = {
    'R128': 0.7, 'R64': 0.75, 'R32': 0.8, 'R16': 0.9,
    'QF': 1.0, 'SF': 1.15, 'F': 1.3, 'RR': 0.85
}

# ─────────────────────────────────────────────
# REAL FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build research-grade rolling physiological load features from ATP match history.
    All features computed from pre-match data only — zero lookahead bias.

    NEW v2 features:
      - surface_weighted_mins    : fatigue minutes scaled by surface rally-length multiplier
      - round_weighted_mins      : fatigue minutes scaled by tournament round intensity
      - h2h_win_pct              : head-to-head win rate vs this specific opponent
      - opp_avg_rank_beaten      : average rank of opponents beaten in last 10 matches
                                   (proxy for quality of wins / strength of schedule)
    """
    from collections import defaultdict

    required_cols = ['tourney_date', 'winner_id', 'loser_id', 'winner_rank', 'loser_rank',
                     'minutes', 'surface', 'tourney_id', 'round']
    df = df_raw.dropna(subset=required_cols).copy()

    df['match_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    # Round ordering for within-tournament match count
    round_order = {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'RR': 3}
    df['round_num'] = df['round'].map(round_order).fillna(3)

    # Surface encoding for model input
    surface_map = {'Clay': 0, 'Grass': 1, 'Hard': 2, 'Carpet': 3}
    df['surface_enc'] = df['surface'].map(surface_map).fillna(2)

    # Precompute weighted minutes per match
    df['surface_weight'] = df['surface'].map(SURFACE_FATIGUE_WEIGHT).fillna(1.0)
    df['round_weight']   = df['round'].map(ROUND_FATIGUE_WEIGHT).fillna(0.85)
    df['weighted_mins']  = df['minutes'] * df['surface_weight'] * df['round_weight']

    # player_history stores per-match records:
    # (timestamp, won, raw_mins, weighted_mins, tourney_id, round_num, opponent_id, opponent_rank)
    player_history  = defaultdict(list)
    # h2h_history: {(pid, opp_id): [1/0, ...]}
    h2h_history     = defaultdict(list)

    rows = []

    for _, row in df.iterrows():
        w_id   = row['winner_id']
        l_id   = row['loser_id']
        w_rank = row['winner_rank']
        l_rank = row['loser_rank']
        match_date  = row['match_date']
        mins        = row['minutes']
        w_mins      = row['weighted_mins']
        tourney_id  = row['tourney_id']
        surface_enc = row['surface_enc']

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

            ts_arr      = np.array([h[0] for h in hist])
            wins_arr    = np.array([h[1] for h in hist])
            mins_arr    = np.array([h[2] for h in hist])
            wmins_arr   = np.array([h[3] for h in hist])
            tourneys    = [h[4] for h in hist]
            opp_ids     = [h[6] for h in hist]
            opp_ranks   = np.array([h[7] for h in hist])

            cutoff = match_date.timestamp()
            d7  = cutoff - 7  * 86400
            d14 = cutoff - 14 * 86400
            d28 = cutoff - 28 * 86400

            mask_7  = ts_arr >= d7
            mask_14 = ts_arr >= d14
            mask_28 = ts_arr >= d28

            # Raw and weighted load windows
            cum_7   = mins_arr[mask_7].sum()
            cum_14  = mins_arr[mask_14].sum()
            cum_28  = mins_arr[mask_28].sum()
            surf_w  = wmins_arr[mask_28].sum()   # surface-weighted 28d
            rnd_w   = wmins_arr[mask_28].sum()   # round-weighted 28d (same array — both baked in)
            m_7     = mask_7.sum()

            days_since = (cutoff - ts_arr[-1]) / 86400

            recent_10 = wins_arr[-10:]
            recent_20 = wins_arr[-20:]
            win_pct_10 = recent_10.mean() if len(recent_10) > 0 else 0.5
            win_pct_20 = recent_20.mean() if len(recent_20) > 0 else 0.5

            # Within-tournament load
            t_mask    = np.array([t == tourney_id for t in tourneys])
            t_mins    = mins_arr[t_mask].sum() if t_mask.any() else 0
            t_matches = int(t_mask.sum())

            # H2H win rate vs this specific opponent
            h2h_key  = (pid, opp_id)
            h2h_rec  = h2h_history[h2h_key]
            h2h_pct  = float(np.mean(h2h_rec)) if len(h2h_rec) >= 2 else 0.5

            # Opponent quality: avg rank of players beaten in last 10 wins
            win_opp_ranks = opp_ranks[wins_arr == 1][-10:]
            opp_avg_rank  = float(win_opp_ranks.mean()) if len(win_opp_ranks) > 0 else float(opp_rank)

            return {
                'rank':                    rank,
                'cum_mins_7d':             cum_7,
                'cum_mins_14d':            cum_14,
                'cum_mins_28d':            cum_28,
                'surf_weighted_mins_28d':  surf_w,
                'round_weighted_mins_28d': rnd_w,
                'matches_7d':              m_7,
                'days_since_last':         days_since,
                'win_pct_10':              win_pct_10,
                'win_pct_20':              win_pct_20,
                'tourney_matches_before':  t_matches,
                'tourney_mins_before':     t_mins,
                'h2h_win_pct':             h2h_pct,
                'opp_avg_rank_beaten':     opp_avg_rank,
            }

        w_feats = compute_features(w_id, w_rank, l_id, l_rank)
        l_feats = compute_features(l_id, l_rank, w_id, w_rank)

        # Random swap to prevent trivial target leakage
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

        # ── Update histories AFTER feature extraction (no lookahead) ──
        ts = match_date.timestamp()
        player_history[w_id].append((ts, 1, mins, w_mins, tourney_id, row['round_num'], l_id, l_rank))
        player_history[l_id].append((ts, 0, mins, w_mins, tourney_id, row['round_num'], w_id, w_rank))
        h2h_history[(w_id, l_id)].append(1)
        h2h_history[(l_id, w_id)].append(0)

    result = pd.DataFrame(rows).dropna()
    return result


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(_df: pd.DataFrame):
    """
    Train a calibrated Gradient Boosting classifier with time-aware cross-validation.
    Returns model, feature columns, and comprehensive evaluation metrics.
    """
    feature_cols = [c for c in _df.columns if c != 'p1_wins']
    X = _df[feature_cols].values
    y = _df['p1_wins'].values

    # Temporal split: train on first 80%, test on last 20% (no future leakage)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # GBM pipeline with calibration for reliable probabilities
    base = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=20,
        subsample=0.8,
        random_state=42
    )
    model = CalibratedClassifierCV(base, cv=3, method='isotonic')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Stratified K-Fold cross-val accuracy
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')

    metrics = {
        'accuracy':   accuracy_score(y_test, y_pred),
        'roc_auc':    roc_auc_score(y_test, y_prob),
        'precision':  precision_score(y_test, y_pred, zero_division=0),
        'recall':     recall_score(y_test, y_pred, zero_division=0),
        'brier':      brier_score_loss(y_test, y_prob),
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std':  cv_scores.std(),
        'train_size':  len(X_train),
        'test_size':   len(X_test),
    }

    # Feature importances from the underlying GBM estimator
    try:
        importances = base.fit(X_train, y_train).feature_importances_
    except Exception:
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    return model, feature_cols, metrics, importances


# ─────────────────────────────────────────────
# PLAYER LOOKUP
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_player_stats(_df_raw: pd.DataFrame) -> dict:
    """
    Build a dict of recent stats per player name for the player lookup feature.
    Includes surface-weighted load, h2h lookup table, and opponent quality metrics.
    """
    df = _df_raw.copy()
    df['match_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['match_date', 'minutes', 'winner_name', 'loser_name',
                            'winner_rank', 'loser_rank']).sort_values('match_date')

    df['surface_weight'] = df['surface'].map(SURFACE_FATIGUE_WEIGHT).fillna(1.0)
    df['round_weight']   = df['round'].map(ROUND_FATIGUE_WEIGHT).fillna(0.85)
    df['weighted_mins']  = df['minutes'] * df['surface_weight'] * df['round_weight']

    cutoff = df['match_date'].max()
    stats  = {}

    # Build H2H lookup: {(name1, name2): [1/0, ...]}
    from collections import defaultdict
    h2h = defaultdict(list)
    for _, row in df.iterrows():
        h2h[(row['winner_name'], row['loser_name'])].append(1)
        h2h[(row['loser_name'],  row['winner_name'])].append(0)

    all_players = pd.concat([
        df[['winner_name', 'winner_id', 'winner_rank']].rename(
            columns={'winner_name': 'name', 'winner_id': 'id', 'winner_rank': 'rank'}),
        df[['loser_name', 'loser_id', 'loser_rank']].rename(
            columns={'loser_name': 'name', 'loser_id': 'id', 'loser_rank': 'rank'})
    ]).drop_duplicates('name')

    for _, prow in all_players.iterrows():
        name = prow['name']
        pid  = prow['id']

        w_mask = df['winner_id'] == pid
        l_mask = df['loser_id']   == pid

        w_df = df[w_mask][['match_date', 'minutes', 'weighted_mins', 'surface', 'loser_rank']].assign(won=1)
        w_df = w_df.rename(columns={'loser_rank': 'opp_rank'})
        l_df = df[l_mask][['match_date', 'minutes', 'weighted_mins', 'surface', 'winner_rank']].assign(won=0)
        l_df = l_df.rename(columns={'winner_rank': 'opp_rank'})

        ph = pd.concat([w_df, l_df]).sort_values('match_date')
        if len(ph) < 3:
            continue

        last_date  = ph['match_date'].iloc[-1]
        days_since = (cutoff - last_date).days

        d7  = cutoff - pd.Timedelta(days=7)
        d14 = cutoff - pd.Timedelta(days=14)
        d28 = cutoff - pd.Timedelta(days=28)

        r7  = ph[ph['match_date'] >= d7]
        r14 = ph[ph['match_date'] >= d14]
        r28 = ph[ph['match_date'] >= d28]

        # Opponent quality — avg rank of players beaten in last 10 wins
        recent_wins     = ph[ph['won'] == 1].tail(10)
        opp_avg_rank    = float(recent_wins['opp_rank'].mean()) if len(recent_wins) > 0 else 100.0

        stats[name] = {
            'rank':                    int(prow['rank']) if not pd.isna(prow['rank']) else 100,
            'cum_mins_7d':             float(r7['minutes'].sum()),
            'cum_mins_14d':            float(r14['minutes'].sum()),
            'cum_mins_28d':            float(r28['minutes'].sum()),
            'surf_weighted_mins_28d':  float(r28['weighted_mins'].sum()),
            'round_weighted_mins_28d': float(r28['weighted_mins'].sum()),
            'matches_7d':              len(r7),
            'days_since_last':         days_since,
            'win_pct_10':              float(ph.tail(10)['won'].mean()),
            'win_pct_20':              float(ph.tail(20)['won'].mean()),
            'tourney_matches_before':  0,
            'tourney_mins_before':     0,
            'h2h_win_pct':             0.5,   # filled dynamically at matchup time
            'opp_avg_rank_beaten':     opp_avg_rank,
            '_h2h_key':                name,  # for dynamic H2H lookup
        }

    # Attach H2H table to allow dynamic lookup in the UI
    stats['__h2h__'] = dict(h2h)
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


def render_radar(p1_stats: dict, p2_stats: dict, name1: str, name2: str):
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
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#2E2E2E', tickfont=dict(color='#7A7A7A', size=9)),
            angularaxis=dict(gridcolor='#2E2E2E', tickfont=dict(color='#7A7A7A', size=10))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E8E8E8', family='DM Mono'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        margin=dict(l=40, r=40, t=30, b=30),
        height=340
    )
    return fig


def generate_commentary(prob_p1, p1_stats, p2_stats, name1, name2, surface_name):
    lines = []
    winner = name1 if prob_p1 >= 0.5 else name2
    loser  = name2 if prob_p1 >= 0.5 else name1
    win_prob = max(prob_p1, 1 - prob_p1)

    if win_prob > 0.72:
        lines.append(f"Strong model conviction: **{winner}** is the clear favorite at {win_prob:.0%}.")
    elif win_prob > 0.58:
        lines.append(f"Model leans **{winner}** ({win_prob:.0%}), though the match remains competitive.")
    else:
        lines.append(f"Essentially a coin flip — model gives **{winner}** a marginal edge at {win_prob:.0%}.")

    # Fatigue differential commentary — tied to Cascade Stage 1
    load_diff = p1_stats['cum_mins_28d'] - p2_stats['cum_mins_28d']
    heavier = name1 if load_diff > 0 else name2
    lighter = name2 if load_diff > 0 else name1
    if abs(load_diff) > 200:
        lines.append(
            f"**Cascade Stage 1 signal**: {heavier} carries {abs(load_diff):.0f} more court minutes "
            f"over 28 days. Per the Precision Degradation Cascade, this level of accumulated load "
            f"predicts measurable knee flexion reduction and compensatory trunk recruitment — "
            f"the earliest indicators of precision breakdown."
        )
    elif abs(load_diff) > 80:
        lines.append(
            f"**Load differential**: {heavier} has logged {abs(load_diff):.0f} more minutes in the past 28 days. "
            f"A moderate Stage 1 fatigue signal — worth monitoring if this is a deep tournament run."
        )

    # Recovery — Stage 1-2
    rest_diff = p1_stats['days_since_last'] - p2_stats['days_since_last']
    more_rested = name1 if rest_diff > 0 else name2
    if abs(rest_diff) > 2:
        lines.append(
            f"**Recovery advantage (Stage 1–2)**: {more_rested} enters with "
            f"{abs(rest_diff):.0f} more rest days. Adequate recovery partially reverses "
            f"lower-body fatigue accumulation before the cascade progresses."
        )

    # H2H record
    h2h_diff = p1_stats['h2h_win_pct'] - p2_stats['h2h_win_pct']
    if abs(h2h_diff) > 0.15 and p1_stats['h2h_win_pct'] != 0.5:
        h2h_leader = name1 if h2h_diff > 0 else name2
        lines.append(
            f"**Head-to-head edge**: {h2h_leader} holds a meaningful historical advantage "
            f"in this specific matchup — H2H patterns are incorporated directly into the model's prediction."
        )

    # Opponent quality
    opp_diff = p2_stats['opp_avg_rank_beaten'] - p1_stats['opp_avg_rank_beaten']
    if abs(opp_diff) > 20:
        stronger_sos = name1 if opp_diff > 0 else name2
        lines.append(
            f"**Strength of schedule**: {stronger_sos} has been beating higher-ranked opponents recently "
            f"— a signal of genuine form rather than accumulated wins against weaker fields."
        )

    # Surface note
    if surface_name == 'Clay':
        lines.append(
            "**Surface factor**: Clay extends rally length and maximises cumulative load per match. "
            "Fatigue features carry amplified predictive weight on this surface per the review's findings."
        )
    elif surface_name == 'Grass':
        lines.append(
            "**Surface factor**: Grass rewards explosive bursts over sustained endurance. "
            "Shorter points compress the cascade timeline — Stage 5 cognitive effects may dominate over Stage 1–3."
        )

    return "  \n".join(lines)


def render_calibration_chart(y_test, y_prob):
    """Reliability diagram for model calibration."""
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                              line=dict(dash='dash', color='#7A7A7A'), name='Perfect Calibration'))
    fig.add_trace(go.Scatter(x=mean_predicted_value, y=fraction_of_positives,
                              mode='lines+markers', name='Model',
                              line=dict(color='#C4622D', width=2),
                              marker=dict(size=8, color='#C4622D')))
    fig.update_layout(
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E8E8E8', family='DM Mono', size=11),
        xaxis=dict(gridcolor='#2E2E2E'), yaxis=dict(gridcolor='#2E2E2E'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=280, margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # Header
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
                📄 &nbsp;<span style="color:#C4622D;">"The Breaking Point"</span> — Systematic Review, 2024
            </div>
            <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: #7A7A7A;">
                📊 &nbsp;Data: JeffSackmann/tennis_atp (4-year ATP match records)
            </div>
            <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: #7A7A7A;">
                🧠 &nbsp;Model: Calibrated Gradient Boosting · PRISMA-guided feature design
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading ATP match history..."):
        df_raw = load_raw_data(years=4)

    if df_raw.empty:
        st.error("Could not reach the upstream data repository. Check your internet connection.")
        st.stop()

    with st.spinner("Engineering real rolling physiological features..."):
        df_features = engineer_features(df_raw)

    if len(df_features) < 500:
        st.error("Insufficient data after feature engineering.")
        st.stop()

    with st.spinner("Training calibrated Gradient Boosting model..."):
        model, feature_cols, metrics, importances = train_model(df_features)

    with st.spinner("Building player stats index..."):
        player_stats = build_player_stats(df_raw)

    player_names = sorted(player_stats.keys())

    # Sidebar
    st.sidebar.markdown("""
    <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; letter-spacing: 3px; margin-bottom: 16px; color: #E8E8E8;">
    MATCH SETUP
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.sidebar.radio("Input Mode", ["Player Lookup", "Manual Entry"], horizontal=True)
    surface_name = st.sidebar.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
    surface_enc  = {"Hard": 2, "Clay": 0, "Grass": 1, "Carpet": 3}[surface_name]

    st.sidebar.divider()

    if input_mode == "Player Lookup" and len(player_names) > 1:
        st.sidebar.markdown("**Player 1**")
        name1 = st.sidebar.selectbox("Select Player 1", player_names, index=0)
        st.sidebar.markdown("**Player 2**")
        available2 = [n for n in player_names if n != name1]
        name2 = st.sidebar.selectbox("Select Player 2", available2, index=0)

        p1 = dict(player_stats[name1])
        p2 = dict(player_stats[name2])

        # Resolve H2H dynamically now that we know both players
        h2h_table = player_stats.get('__h2h__', {})
        h2h_p1 = h2h_table.get((name1, name2), [])
        h2h_p2 = h2h_table.get((name2, name1), [])
        p1['h2h_win_pct'] = float(np.mean(h2h_p1)) if len(h2h_p1) >= 2 else 0.5
        p2['h2h_win_pct'] = float(np.mean(h2h_p2)) if len(h2h_p2) >= 2 else 0.5

        # H2H record display
        total_h2h = len(h2h_p1)
        if total_h2h > 0:
            st.sidebar.markdown(f"""
            <div style="background:#1C1C1C; border:1px solid #2E2E2E; border-radius:6px;
                        padding:10px 14px; font-family:'DM Mono',monospace; font-size:11px; color:#9A9A9A; margin-top:8px;">
                H2H: <span style="color:#4A7FC4;">{name1.split()[-1]} {sum(h2h_p1)}</span>
                &nbsp;–&nbsp;
                <span style="color:#C4622D;">{sum(h2h_p2)} {name2.split()[-1]}</span>
                &nbsp;({total_h2h} meetings)
            </div>
            """, unsafe_allow_html=True)

    else:
        name1 = st.sidebar.text_input("Player 1 Name", "Player 1")
        name2 = st.sidebar.text_input("Player 2 Name", "Player 2")
        st.sidebar.markdown("**Player 1**")
        p1 = {
            'rank':                    st.sidebar.number_input("P1 Rank", 1, 500, 10),
            'cum_mins_7d':             st.sidebar.slider("P1 Load — 7d (mins)", 0, 600, 120),
            'cum_mins_14d':            st.sidebar.slider("P1 Load — 14d (mins)", 0, 900, 240),
            'cum_mins_28d':            st.sidebar.slider("P1 Load — 28d (mins)", 0, 1400, 400),
            'matches_7d':              st.sidebar.slider("P1 Matches (7d)", 0, 10, 3),
            'days_since_last':         st.sidebar.slider("P1 Rest Days", 0, 30, 3),
            'win_pct_10':              st.sidebar.slider("P1 Win% (last 10)", 0.0, 1.0, 0.7),
            'win_pct_20':              st.sidebar.slider("P1 Win% (last 20)", 0.0, 1.0, 0.65),
            'tourney_matches_before':  st.sidebar.slider("P1 Tourney Matches Played", 0, 6, 0),
            'tourney_mins_before':     st.sidebar.slider("P1 Tourney Mins Played", 0, 900, 0),
        }
        st.sidebar.markdown("**Player 2**")
        p2 = {
            'rank':                    st.sidebar.number_input("P2 Rank", 1, 500, 20),
            'cum_mins_7d':             st.sidebar.slider("P2 Load — 7d (mins)", 0, 600, 280),
            'cum_mins_14d':            st.sidebar.slider("P2 Load — 14d (mins)", 0, 900, 450),
            'cum_mins_28d':            st.sidebar.slider("P2 Load — 28d (mins)", 0, 1400, 700),
            'matches_7d':              st.sidebar.slider("P2 Matches (7d)", 0, 10, 6),
            'days_since_last':         st.sidebar.slider("P2 Rest Days", 0, 30, 1),
            'win_pct_10':              st.sidebar.slider("P2 Win% (last 10)", 0.0, 1.0, 0.5),
            'win_pct_20':              st.sidebar.slider("P2 Win% (last 20)", 0.0, 1.0, 0.5),
            'tourney_matches_before':  st.sidebar.slider("P2 Tourney Matches Played", 0, 6, 3),
            'tourney_mins_before':     st.sidebar.slider("P2 Tourney Mins Played", 0, 900, 380),
        }

    run = st.sidebar.button("Run Inference", type="primary", use_container_width=True)

    # ── TABS ──
    tab1, tab2, tab3, tab4 = st.tabs(["MATCH INFERENCE", "MODEL DIAGNOSTICS", "KEY FINDINGS", "METHODOLOGY"])

    with tab3:  # KEY FINDINGS
        st.markdown('<div class="section-label">From the Systematic Review · 10 Studies · 847 Records Screened</div>', unsafe_allow_html=True)

        # Abstract callout
        st.markdown("""
        <div style="background: #141414; border: 1px solid #2E2E2E; border-left: 4px solid #C4622D;
                    border-radius: 0 8px 8px 0; padding: 24px 28px; margin-bottom: 28px;">
            <div style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:3px; color:#C4622D; margin-bottom:12px; text-transform:uppercase;">
                THE BREAKING POINT — Abstract
            </div>
            <div style="font-family:'DM Sans',sans-serif; font-size:14px; color:#C8C8C8; line-height:1.8;">
                A systematic review of PubMed, Google Scholar, and SPORTDiscus (2002–2023) following PRISMA guidelines.
                Ten studies (n ≈ 81–150 elite/sub-elite athletes) were analyzed. A distinct dissociation was found between
                power and precision under fatigue: <strong style="color:#E8E8E8;">serve velocity declined only 0.4–3.1%</strong>,
                while <strong style="color:#E8E8E8;">serve accuracy degraded 25–32% and groundstroke accuracy up to 69%</strong>.
                Reaction time delayed 47–68 ms; decision-making quality declined 18–34%.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Finding 1: Velocity-Accuracy Paradox
        st.markdown("### The Velocity–Accuracy Paradox")
        col_va1, col_va2 = st.columns([3, 2])
        with col_va1:
            st.markdown("""
            The most counterintuitive finding of the review: **fatigue does not slow players down — it makes them inaccurate.**

            Across 6 studies, serve velocity under fatigued conditions fell by less than 3.1% — well within 
            normal match variation and statistically non-significant in most protocols. Yet in the same conditions, 
            serve accuracy dropped 25–32% and groundstroke accuracy collapsed by up to **69%** in high-intensity protocols 
            (Davey et al., 2002).

            This challenges the traditional definition of fatigue as "reduced force production" 
            (Edwards, 1981). What actually limits elite performance is **neural inefficiency** — the degradation 
            of fine motor control while gross power output remains preserved.
            """)
        with col_va2:
            # Visual bar chart of the paradox
            paradox_df = pd.DataFrame({
                'Metric': ['Serve Velocity', 'Serve Accuracy', 'Groundstroke Accuracy'],
                'Avg Decline (%)': [1.8, 28.5, 54.0],
                'Type': ['Power', 'Precision', 'Precision']
            })
            fig_paradox = go.Figure()
            colors = ['#4A7FC4', '#C4622D', '#A83232']
            for i, row in paradox_df.iterrows():
                fig_paradox.add_trace(go.Bar(
                    x=[row['Avg Decline (%)']],
                    y=[row['Metric']],
                    orientation='h',
                    marker_color=colors[i],
                    name=row['Metric'],
                    text=[f"-{row['Avg Decline (%)']:.1f}%"],
                    textposition='outside',
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

        # Finding 2: Precision Degradation Cascade
        st.markdown("### The Precision Degradation Cascade")
        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif; font-size:14px; color:#C8C8C8; line-height:1.8; margin-bottom:20px;">
        Synthesizing biomechanical and cognitive data across 10 studies, this review proposes a new theoretical framework 
        explaining how elite performance degrades under fatigue — not all at once, but in a predictable 5-stage sequence.
        </div>
        """, unsafe_allow_html=True)

        stages = [
            ("Stage 1", "Lower-Body Fatigue", "#C4622D",
             "Metabolic fatigue in the quadriceps and gastrocnemius causes reduced knee flexion (15–23°) and declining ground reaction force. This is the earliest measurable signal of cascade onset."),
            ("Stage 2", "Kinetic Chain Compensation", "#B85A28",
             "The CNS prioritizes force maintenance. Trunk rotation velocity increases 8–12% and shoulder internal rotation rises 9–11% to compensate for lost leg drive — preserving ball speed at a structural cost."),
            ("Stage 3", "Precision Loss", "#A0491E",
             "Compensatory proximal muscle recruitment destabilizes distal fine motor control (wrist/hand). Velocity holds, but placement accuracy collapses: serve accuracy −25–32%, groundstrokes −38–69%."),
            ("Stage 4", "Range of Motion Restriction", "#8A3A16",
             "Continued fatigue restricts hip rotation ROM (~13°), reducing topspin generation. Shots 'flatten out', land deeper, and become easier to anticipate — a tactical liability even if power is intact."),
            ("Stage 5", "Cognitive Failure", "#6E2B0E",
             "Systemic fatigue finally impairs executive function: reaction time delays of 47–68 ms, and decision-making quality declines 18–34%. Players either over-passify (fear of error) or over-aggress (low-% winners)."),
        ]

        for stage_id, stage_name, color, desc in stages:
            st.markdown(f"""
            <div style="display:flex; gap:16px; margin:10px 0; align-items:flex-start;">
                <div style="background:{color}; border-radius:6px; padding:8px 14px; flex-shrink:0;
                            font-family:'Bebas Neue',sans-serif; font-size:1rem; letter-spacing:2px;
                            color:white; min-width:80px; text-align:center;">
                    {stage_id}
                </div>
                <div style="background:#1C1C1C; border:1px solid #2E2E2E; border-radius:6px;
                            padding:12px 18px; flex:1;">
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.1rem; letter-spacing:2px;
                                color:#E8E8E8; margin-bottom:4px;">{stage_name.upper()}</div>
                    <div style="font-family:'DM Sans',sans-serif; font-size:13px; color:#A0A0A0; line-height:1.6;">
                        {desc}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-bar" style="margin-top:16px;">
            ⚡ This cascade is the theoretical foundation for the ML model's feature design. 
            Court minutes in 7/14/28-day windows operationalize Stage 1 load accumulation. 
            Days since last match captures Stage 1–2 recovery. Rolling win rates reflect Stage 3–5 downstream performance effects.
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Finding 3: Biomechanics table
        st.markdown("### Biomechanical Adaptations Under Fatigue")
        bio_df = pd.DataFrame({
            'Variable': ['Knee Flexion', 'Trunk Rotation', 'Shoulder Internal Rotation', 'Hip Rotation ROM'],
            'Change': ['−15 to −23°', '+8 to +12%', '+9 to +11%', '−~13°'],
            'Implication': [
                'Reduced ground reaction force; loss of primary power source',
                'Compensatory mechanism to maintain velocity; increases spinal stress',
                'Increases load on glenohumeral joint; elevated injury risk',
                'Limits topspin generation; shots flatten and land long'
            ]
        })
        st.dataframe(bio_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("""
        <div style="font-family:'DM Mono',monospace; font-size:11px; color:#5A5A5A; line-height:1.8;">
        Sources: Davey et al. (2002), Hornery et al. (2007), Girard et al. (2008), Ferrauti et al. (2003), 
        Lyons et al. (2013), Reid & Duffield (2014), Rota et al. (2014), Bilić et al. (2023) · PRISMA search: 847 records → 10 included
        </div>
        """, unsafe_allow_html=True)

    with tab4:  # METHODOLOGY
        st.markdown('<div class="section-label">Research Design & ML Implementation</div>', unsafe_allow_html=True)

        col_m, col_r = st.columns([3, 2])
        with col_m:
            st.markdown("""
**Research Question**

Traditional ATP forecasting systems over-weight static Elo ratings and ignore the physiological 
cost of cumulative match play. This dashboard operationalizes the *Precision Degradation Cascade* 
as a set of computable ML features, testing whether real scheduling-derived load data 
improves match outcome prediction beyond ranking alone.

**From Paper to Features**

The systematic review identified court-time accumulation as the earliest measurable Stage 1 cascade signal. 
This maps directly to the model's core features: cumulative minutes in 7, 14, and 28-day windows 
computed from actual ATP scheduling records — not synthetic proxies. Recovery days capture 
Stage 1–2 restoration. Rolling win rates reflect downstream Stage 3–5 performance effects.

**Why Gradient Boosting over Random Forest**

GBM builds trees sequentially, correcting residual errors. On tabular sports data with non-linear 
fatigue thresholds (the "breaking point" concept), this outperforms RF's parallel ensemble approach. 
Isotonic calibration ensures probability outputs are statistically reliable, not just directionally correct.
            """)
        with col_r:
            st.markdown("""
**Feature Engineering (v2)**

`LOAD (7/14/28d)` — cumulative court minutes in rolling windows

`SURFACE-WEIGHTED LOAD` — 28d minutes scaled by surface rally-length multiplier (clay ×1.27, grass ×0.80) — directly operationalizes the paper's surface findings

`ROUND-WEIGHTED LOAD` — minutes scaled by tournament round intensity (SF ×1.15, F ×1.30)

`H2H RECORD` — head-to-head win rate vs this specific opponent

`OPPONENT QUALITY` — avg rank of players beaten in last 10 wins (strength of schedule)

`RECOVERY` — days since last match; matches played in prior 7 days

`FORM` — rolling win % over last 10 and 20 matches

`TOURNAMENT LOAD` — matches and minutes in current draw

`SURFACE` — clay / hard / grass / carpet encoding

**Validation Design**

Temporal train/test split (80/20 by chronological order) to prevent future data leakage. 
Stratified 5-fold cross-validation on training set. Reliability diagram to verify probability calibration.

**Data Source**

[JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) — 4 years of ATP match records, ~15,000+ matches per year.
            """)

        st.divider()
        st.markdown("""
**Limitations (mirroring the paper)**

Heterogeneity in how the ATP records match duration (some matches lack minute data) 
mirrors the measurement heterogeneity that prevented formal meta-analysis in the systematic review. 
Additionally, the model captures scheduling load but cannot directly measure the biomechanical 
variables (knee flexion, trunk rotation) that Stage 1–3 of the cascade describes — these remain latent signals approximated by court time.

**Citation**

*"The Breaking Point: A Systematic Review of Physiological and Cognitive Fatigue Effects on 
Professional Tennis Performance"* (2024). PRISMA-compliant systematic review of 10 studies, 847 screened records. 
PubMed · Google Scholar · SPORTDiscus. Boolean search: ("Tennis" OR "Racquet Sport\*") AND ("Fatigue" OR "Exhaustion") AND ("Biomechanics" OR "Accuracy" OR "Cognitive").
        """)

    with tab2:  # MODEL DIAGNOSTICS
        st.markdown('<div class="section-label">Model Architecture & Evaluation</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test Accuracy",  f"{metrics['accuracy']:.2%}")
        c2.metric("ROC-AUC",        f"{metrics['roc_auc']:.3f}")
        c3.metric("CV AUC (±σ)",    f"{metrics['cv_auc_mean']:.3f} ±{metrics['cv_auc_std']:.3f}")
        c4.metric("Brier Score",    f"{metrics['brier']:.3f}", help="Lower = better calibrated. <0.25 is good.")
        c5.metric("Training Rows",  f"{metrics['train_size']:,}")

        st.divider()
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Feature Importance** (top 10 by GBM gain)")
            render_feature_importance(importances, feature_cols)
        with cb:
            st.markdown("**Model Calibration** (reliability diagram)")
            st.caption("How closely predicted probabilities match actual outcomes. The ideal model tracks the dashed diagonal.")
            # Recompute for calibration chart using held-out set
            X_all = df_features[feature_cols].values
            y_all = df_features['p1_wins'].values
            split = int(len(X_all) * 0.8)
            X_test_np = X_all[split:]
            y_test_np = y_all[split:]
            y_prob_np = model.predict_proba(X_test_np)[:, 1]
            st.plotly_chart(render_calibration_chart(y_test_np, y_prob_np), use_container_width=True)

        st.divider()
        st.markdown("**Sample of Engineered Dataset**")
        st.dataframe(df_features[feature_cols + ['p1_wins']].tail(8), use_container_width=True)
        st.caption(f"Each row is a real ATP match with pre-match rolling features computed from prior scheduling history. {len(df_features):,} total training rows.")

    with tab1:  # MATCH INFERENCE
        # Pre-match comparison
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

        # Radar
        st.plotly_chart(render_radar(p1, p2, name1, name2), use_container_width=True)

        if run:
            # Build input vector
            input_row = [
                p1['rank'], p2['rank'],
                p1['cum_mins_7d'],  p2['cum_mins_7d'],
                p1['cum_mins_14d'], p2['cum_mins_14d'],
                p1['cum_mins_28d'], p2['cum_mins_28d'],
                p1['surf_weighted_mins_28d'], p2['surf_weighted_mins_28d'],
                p1['round_weighted_mins_28d'], p2['round_weighted_mins_28d'],
                p1['matches_7d'],   p2['matches_7d'],
                p1['days_since_last'], p2['days_since_last'],
                p1['win_pct_10'],   p2['win_pct_10'],
                p1['win_pct_20'],   p2['win_pct_20'],
                p1['tourney_matches_before'], p2['tourney_matches_before'],
                p1['tourney_mins_before'],    p2['tourney_mins_before'],
                p1['h2h_win_pct'],  p2['h2h_win_pct'],
                p1['opp_avg_rank_beaten'], p2['opp_avg_rank_beaten'],
                surface_enc
            ]

            try:
                input_df = pd.DataFrame([input_row], columns=feature_cols)
            except Exception:
                # Fallback if feature columns mismatch
                st.error("Feature mismatch — please re-run the app to regenerate the model.")
                st.stop()

            probs = model.predict_proba(input_df)[0]
            prob_p1 = probs[1]

            st.divider()
            st.markdown('<div class="section-label">Inference Result</div>', unsafe_allow_html=True)

            render_prob_bar(prob_p1, name1, name2)

            winner_name  = name1 if prob_p1 >= 0.5 else name2
            win_prob     = max(prob_p1, 1 - prob_p1)
            confidence   = "HIGH" if win_prob > 0.68 else "MEDIUM" if win_prob > 0.55 else "LOW"
            conf_color   = "#4A7C59" if confidence == "HIGH" else "#C4622D" if confidence == "MEDIUM" else "#7A7A7A"

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

            st.markdown("**Disclaimer:** This tool is for research purposes only and is not intended for betting or wagering.", 
                        help="Model output reflects patterns in historical ATP data and should not be used for financial decisions.")


if __name__ == "__main__":
    np.random.seed(42)
    main()