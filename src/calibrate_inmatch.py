"""
Chunk 4A: Check and fix calibration of the in-match model.
A reliability diagram tells us whether predicted probabilities match reality.
"""
import io
import socket
import urllib.error
import urllib.request

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/refs/heads/master/charting-m-points-2020s.csv"
BO5_TOURNEYS = ['Australian_Open', 'Roland_Garros', 'Wimbledon', 'US_Open']

def read_remote_csv(url: str, label: str, timeout: int = 20) -> pd.DataFrame:
    """Download a remote CSV with timeout, HTTP, and empty-response handling."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{label} returned HTTP {e.code}.") from e
    except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
        raise RuntimeError(f"{label} request failed or timed out.") from e

    if not payload:
        raise RuntimeError(f"{label} returned an empty response.")

    try:
        df = pd.read_csv(io.BytesIO(payload), low_memory=False)
    except pd.errors.EmptyDataError as e:
        raise RuntimeError(f"{label} returned an empty CSV.") from e

    if df.empty:
        raise RuntimeError(f"{label} did not contain any rows.")
    return df


def infer_best_of_5(match_id):
    """Infer best-of-5 format from Grand Slam match IDs."""
    if not isinstance(match_id, str):
        return 0
    return int(any(t in match_id for t in BO5_TOURNEYS))

print("Downloading data...")
try:
    df = read_remote_csv(URL, "Match Charting Project point data")
except RuntimeError as e:
    raise SystemExit(f"Could not load point data: {e}") from e

# --- Match winners ---
last_points = df.groupby('match_id').tail(1).copy()
last_points['p1_won_match'] = (last_points['Set1'] > last_points['Set2']).astype(int)
df = df.merge(last_points[['match_id', 'p1_won_match']], on='match_id', how='left')

# --- Features ---
df['set_diff']                 = df['Set1'] - df['Set2']
df['game_diff']                = df['Gm1'] - df['Gm2']
df['is_p1_serving']            = (df['Svr'] == 1).astype(int)
df['pt_number']                = df['Pt']
df['total_sets']               = df['Set1'] + df['Set2']
df['is_tiebreak']              = ((df['Gm1'] == 6) & (df['Gm2'] == 6)).astype(int)
df['game_total']               = df['Gm1'] + df['Gm2']
df['score_advantage_strength'] = df['set_diff'].abs() * 2 + df['game_diff'].abs()
df['is_best_of_5']             = df['match_id'].apply(infer_best_of_5)
df['max_sets_to_win']          = df['is_best_of_5'].apply(lambda x: 3 if x else 2)
df['is_decisive_set']          = (df['total_sets'] == (df['max_sets_to_win'] * 2 - 2)).astype(int)
df['target']                   = df['p1_won_match']

feature_cols = ['set_diff', 'game_diff', 'is_p1_serving', 'pt_number',
                'total_sets', 'is_tiebreak', 'game_total',
                'score_advantage_strength', 'is_best_of_5', 'is_decisive_set']
df = df.dropna(subset=feature_cols + ['target'])

# --- Symmetry augmentation ---
df_swapped = df.copy()
df_swapped['set_diff']      = -df['set_diff']
df_swapped['game_diff']     = -df['game_diff']
df_swapped['is_p1_serving'] = 1 - df['is_p1_serving']
df_swapped['target']        = 1 - df['target']
df_full = pd.concat([df, df_swapped], ignore_index=True)

# --- Split by match ---
all_matches = df_full['match_id'].unique().tolist()
np.random.seed(42)
np.random.shuffle(all_matches)
split_idx = int(len(all_matches) * 0.8)
train_matches = set(all_matches[:split_idx])
train_df = df_full[df_full['match_id'].isin(train_matches)]
test_df  = df_full[~df_full['match_id'].isin(train_matches)]

X_train, y_train = train_df[feature_cols].values, train_df['target'].values
X_test,  y_test  = test_df[feature_cols].values,  test_df['target'].values

# ─────────────────────────────────────────────
# UNCALIBRATED model
# ─────────────────────────────────────────────
print("\nTraining uncalibrated GBM...")
gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                 max_depth=4, subsample=0.8, random_state=42)
gbm.fit(X_train, y_train)
raw_prob = gbm.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────────
# CALIBRATED model (isotonic)
# ─────────────────────────────────────────────
print("Training calibrated GBM (isotonic)... takes a few minutes...")
cal = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                               max_depth=4, subsample=0.8, random_state=42),
    cv=3, method='isotonic'
)
cal.fit(X_train, y_train)
cal_prob = cal.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────────
# Compare
# ─────────────────────────────────────────────
print("\n─── CALIBRATION COMPARISON ───")
print(f"{'Metric':<14}{'Uncalibrated':>14}{'Calibrated':>14}")
print(f"{'AUC':<14}{roc_auc_score(y_test, raw_prob):>14.4f}{roc_auc_score(y_test, cal_prob):>14.4f}")
print(f"{'Brier':<14}{brier_score_loss(y_test, raw_prob):>14.4f}{brier_score_loss(y_test, cal_prob):>14.4f}")
print(f"{'LogLoss':<14}{log_loss(y_test, np.clip(raw_prob,1e-6,1-1e-6)):>14.4f}{log_loss(y_test, np.clip(cal_prob,1e-6,1-1e-6)):>14.4f}")

# Reliability table (text version of the diagram)
print("\n─── RELIABILITY (calibrated model) ───")
print("Predicted bucket → actual P1 win rate (want them to match):")
frac, mean_pred = calibration_curve(y_test, cal_prob, n_bins=10)
for mp, fr in zip(mean_pred, frac):
    gap = abs(mp - fr)
    flag = "✓" if gap < 0.05 else "⚠"
    print(f"  predicted {mp:.2f} → actual {fr:.2f}  (gap {gap:.3f}) {flag}")
