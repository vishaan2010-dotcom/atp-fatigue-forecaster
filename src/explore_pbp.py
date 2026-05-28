import pandas as pd

# Sackmann's Match Charting Project — point-by-point data for men's matches
URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/charting-m-points.csv"

# Load it
df = pd.read_csv(URL, low_memory=False)

# 1. How big is it?
print("Total rows (points):", len(df))
print("Total columns:", len(df.columns))

# 2. What columns exist?
print("\nColumns:")
print(df.columns.tolist())

# 3. Show the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 4. How many unique matches?
print("\nUnique matches:", df['match_id'].nunique())
