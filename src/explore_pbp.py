"""
Explore the Match Charting Project point-by-point data.
Data: Jeff Sackmann's Match Charting Project (https://github.com/JeffSackmann/tennis_MatchChartingProject)
License: CC BY-NC-SA 4.0 (Non-commercial, attribution required)
"""
import pandas as pd

# Men's points data — 2020s decade file.
# The Match Charting Project splits points by decade because the full file is huge.
URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/charting-m-points-2020s.csv"

print("Downloading point-by-point data... (this may take 30-60 seconds)")
df = pd.read_csv(URL, low_memory=False)

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