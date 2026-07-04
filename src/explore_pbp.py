"""
Explore the Match Charting Project point-by-point data.
Data: Jeff Sackmann's Match Charting Project (https://github.com/JeffSackmann/tennis_MatchChartingProject)
License: CC BY-NC-SA 4.0 (Non-commercial, attribution required)
"""
import io
import socket
import urllib.error
import urllib.request

import pandas as pd

# Men's points data, 2020s decade file.
# The Match Charting Project splits points by decade because the full file is huge.
URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/refs/heads/master/charting-m-points-2020s.csv"

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


print("Downloading point-by-point data... (this may take 30-60 seconds)")
try:
    df = read_remote_csv(URL, "Match Charting Project point data")
except RuntimeError as e:
    raise SystemExit(f"Could not load point data: {e}") from e

# 1. How big is it?
print(f"\nTotal rows (points): {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# 2. What columns exist?
print("\nColumns:")
print(df.columns.tolist())

# 3. Show the first 5 rows of a few key columns
print("\nFirst 5 rows of key columns:")
key_cols = [c for c in ['match_id', 'Pt', 'Set1', 'Set2', 'Game1', 'Game2', 'Svr', 'PtWinner'] if c in df.columns]
print(df[key_cols].head())

# 4. How many unique matches?
print(f"\nUnique matches: {df['match_id'].nunique():,}")
