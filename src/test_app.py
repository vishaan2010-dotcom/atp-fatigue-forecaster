import pytest
import pandas as pd
from app import load_and_preprocess_data

def test_data_pipeline():
    """Tests if the data pipeline successfully fetches and engineers the features."""
    df = load_and_preprocess_data()
    
    # 1. Check if it returns a pandas DataFrame
    assert isinstance(df, pd.DataFrame), "Data pipeline failed to return a DataFrame"
    
    # 2. Check if the dataframe actually has data (meaning the Sackmann repo is up)
    assert len(df) > 0, "Data pipeline returned an empty dataset"
    
    # 3. Check if our custom fatigue features were created successfully
    assert 'p1_cumulative_minutes' in df.columns, "Missing engineered fatigue feature for P1"
    assert 'p2_cumulative_minutes' in df.columns, "Missing engineered fatigue feature for P2"