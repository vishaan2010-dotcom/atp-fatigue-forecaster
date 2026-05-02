"""
Tests for ATP Fatigue Forecaster data pipeline.
Validates that the data loading and feature engineering produce a usable dataset.
"""
import pytest
import pandas as pd
from app import load_raw_data, engineer_features


def test_raw_data_loads():
    """Sackmann ATP repo is reachable and returns match data."""
    df = load_raw_data(years=2)  # 2 years is enough for a fast test
    assert isinstance(df, pd.DataFrame), "load_raw_data must return a DataFrame"
    assert len(df) > 0, "Sackmann repo returned an empty dataset"
    # Required raw columns for downstream feature engineering
    for col in ['tourney_date', 'winner_id', 'loser_id', 'minutes', 'surface']:
        assert col in df.columns, f"Raw data missing required column: {col}"


def test_feature_engineering_produces_rolling_features():
    """Feature engineering creates the rolling physiological-load features."""
    df_raw = load_raw_data(years=2)
    df_features = engineer_features(df_raw)

    assert isinstance(df_features, pd.DataFrame)
    assert len(df_features) > 0, "Feature engineering produced empty dataset"

    # Core Stage 1 cascade features must exist for both players
    required_features = [
        'p1_cum_mins_7d',  'p2_cum_mins_7d',
        'p1_cum_mins_14d', 'p2_cum_mins_14d',
        'p1_cum_mins_28d', 'p2_cum_mins_28d',
        'p1_days_since_last', 'p2_days_since_last',
        'p1_h2h_win_pct', 'p2_h2h_win_pct',
        'p1_wins',
    ]
    for feat in required_features:
        assert feat in df_features.columns, f"Missing engineered feature: {feat}"


def test_target_variable_is_balanced():
    """p1_wins should be roughly 50/50 due to random player-side assignment."""
    df_raw = load_raw_data(years=2)
    df_features = engineer_features(df_raw)
    win_rate = df_features['p1_wins'].mean()
    assert 0.40 < win_rate < 0.60, f"Target imbalance: p1 wins {win_rate:.2%} of matches"


def test_no_lookahead_bias():
    """Rolling features at the first match should be empty (no prior history)."""
    df_raw = load_raw_data(years=2)
    df_features = engineer_features(df_raw)
    # Most early matches should have zero or near-zero cumulative minutes
    early_matches = df_features.head(50)
    assert early_matches['p1_cum_mins_28d'].mean() < 500, \
        "Early matches should have low cumulative minutes (no prior history)"
