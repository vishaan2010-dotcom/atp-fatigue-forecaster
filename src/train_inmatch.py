"""
Chunk 3 (v2): Score-state features + GBM vs logistic comparison.
Fixes: TbSet column was not 0/1, derive tiebreak from score (6-6).
"""
import io
import socket
import urllib.error
import urllib.request

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
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


def infer_best_of_5(match_id: str) -> int:
    """Infer best-of-5 format from Grand Slam match IDs."""
    if not isinstance(match_id, str):
        return 0
    return int(any(t in match_id for t in BO5_TOURNEYS))

print("Downloading data...")
try:
    df = read_remote_csv(URL, "Match Charting Project point data")
except RuntimeError as e:
    raise SystemExit(f"Could not load point data: {e}") from e
print(f"Loaded {len(df):,} points from {df['match_id'].nunique():,} matches.")

# ─────────────────────────────────────────────
# STEP 1: Match winners
# ─────────────────────────────────────────────
last_points = df.groupby('match_id').tail(1).copy()
last_points['p1_won_match'] = (last_points['Set1'] > last_points['Set2']).astype(int)
winners = last_points[['match_id', 'p1_won_match']]
df = df.merge(winners, on='match_id', how='left')

# ─────────────────────────────────────────────
# STEP 2: Features (P1-perspective)
# ─────────────────────────────────────────────
print("\nBuilding features...")
df['set_diff']                 = df['Set1'] - df['Set2']
df['game_diff']                = df['Gm1']  - df['Gm2']
df['is_p1_serving']            = (df['Svr'] == 1).astype(int)
df['pt_number']                = df['Pt']
df['total_sets']               = df['Set1'] + df['Set2']

# Tiebreak: derived from score, not the unreliable TbSet column
df['is_tiebreak']              = ((df['Gm1'] == 6) & (df['Gm2'] == 6)).astype(int)

df['game_total']               = df['Gm1'] + df['Gm2']
df['score_advantage_strength'] = df['set_diff'].abs() * 2 + df['game_diff'].abs()
df['is_best_of_5']             = df['match_id'].apply(infer_best_of_5)
df['max_sets_to_win']          = df['is_best_of_5'].apply(lambda x: 3 if x else 2)
df['is_decisive_set']          = (df['total_sets'] == (df['max_sets_to_win'] * 2 - 2)).astype(int)
df['target']                   = df['p1_won_match']

df = df.dropna(subset=['set_diff', 'game_diff', 'is_p1_serving', 'pt_number',
                       'total_sets', 'is_tiebreak', 'game_total',
                       'score_advantage_strength', 'is_best_of_5',
                       'is_decisive_set', 'target'])

print(f"After cleaning: {len(df):,} rows")
print(f"Best-of-5 matches: {df.groupby('match_id')['is_best_of_5'].first().sum():,}")
print(f"Tiebreak points:   {df['is_tiebreak'].sum():,}  (should be ~1-3% of total)")
print(f"Decisive-set pts:  {df['is_decisive_set'].sum():,}")

# ─────────────────────────────────────────────
# STEP 3: Symmetry augmentation
# ─────────────────────────────────────────────
print("\nApplying symmetry augmentation...")
df_swapped = df.copy()
df_swapped['set_diff']      = -df['set_diff']
df_swapped['game_diff']     = -df['game_diff']
df_swapped['is_p1_serving'] = 1 - df['is_p1_serving']
df_swapped['target']        = 1 - df['target']
df_full = pd.concat([df, df_swapped], ignore_index=True)
print(f"After symmetry: {len(df_full):,} rows. Target rate: {df_full['target'].mean():.1%}")

# ─────────────────────────────────────────────
# STEP 4: Train/test split by match
# ─────────────────────────────────────────────
all_matches = df_full['match_id'].unique().tolist()
np.random.seed(42)
np.random.shuffle(all_matches)
split_idx = int(len(all_matches) * 0.8)
train_matches = set(all_matches[:split_idx])
test_matches  = set(all_matches[split_idx:])
train_df = df_full[df_full['match_id'].isin(train_matches)]
test_df  = df_full[df_full['match_id'].isin(test_matches)]
print(f"\nTrain: {len(train_df):,} points  |  Test: {len(test_df):,} points")

# ─────────────────────────────────────────────
# STEP 5: Train + evaluate BOTH models
# ─────────────────────────────────────────────
feature_cols = [
    'set_diff', 'game_diff', 'is_p1_serving', 'pt_number',
    'total_sets', 'is_tiebreak', 'game_total',
    'score_advantage_strength', 'is_best_of_5', 'is_decisive_set',
]
X_train = train_df[feature_cols].values
y_train = train_df['target'].values
X_test  = test_df[feature_cols].values
y_test  = test_df['target'].values

print("\n=== Linear baseline: Logistic Regression ===")
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
lr_prob = lr.predict_proba(X_test)[:, 1]
print(f"AUC: {roc_auc_score(y_test, lr_prob):.4f}   "
      f"Brier: {brier_score_loss(y_test, lr_prob):.4f}   "
      f"LogLoss: {log_loss(y_test, np.clip(lr_prob, 1e-6, 1-1e-6)):.4f}")

print("\n=== Non-linear model: Gradient Boosting ===")
print("(training takes 2-3 minutes...)")
gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
gbm.fit(X_train, y_train)
gbm_prob = gbm.predict_proba(X_test)[:, 1]
auc   = roc_auc_score(y_test, gbm_prob)
brier = brier_score_loss(y_test, gbm_prob)
ll    = log_loss(y_test, np.clip(gbm_prob, 1e-6, 1-1e-6))

print(f"\n─── GBM RESULTS ───")
print(f"Test AUC:    {auc:.4f}")
print(f"Brier Score: {brier:.4f}")
print(f"Log Loss:    {ll:.4f}")

print(f"\nFeature importance (GBM):")
importances = sorted(zip(gbm.feature_importances_, feature_cols), reverse=True)
for imp, col in importances:
    bar = "█" * int(imp * 100)
    print(f"  {col:30s}: {imp:.4f}  {bar}")

print(f"\nLift over logistic regression: "
      f"+{auc - roc_auc_score(y_test, lr_prob):.4f} AUC")
