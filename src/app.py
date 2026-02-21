import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(page_title="ATP Match Forecaster", layout="wide")

@st.cache_data
def load_data():
    # Simulating data with physiological and cognitive fatigue features
    np.random.seed(42)
    data = pd.DataFrame({
        'player_1_rank': np.random.randint(1, 200, 500),
        'player_2_rank': np.random.randint(1, 200, 500),
        'p1_cumulative_minutes': np.random.randint(120, 600, 500), 
        'p2_cumulative_minutes': np.random.randint(120, 600, 500),
        'p1_recent_win_pct': np.random.uniform(0.3, 0.9, 500), 
        'p2_recent_win_pct': np.random.uniform(0.3, 0.9, 500),
        'p1_wins': np.random.choice([0, 1], 500)
    })
    return data

st.title("🎾 Live ATP Match-Winner Forecaster")
st.markdown("Predictive machine learning model forecasting professional tennis outcomes based on physiological and cognitive fatigue effects.")

# Train the Random Forest Model
df = load_data()
X = df.drop('p1_wins', axis=1)
y = df['p1_wins']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Sidebar UI
st.sidebar.header("Match Setup")
p1_rank = st.sidebar.number_input("Player 1 Rank", min_value=1, max_value=500, value=10)
p2_rank = st.sidebar.number_input("Player 2 Rank", min_value=1, max_value=500, value=15)
p1_fatigue = st.sidebar.slider("P1 Cumulative Minutes Played", 0, 1000, 300)
p2_fatigue = st.sidebar.slider("P2 Cumulative Minutes Played", 0, 1000, 450)
p1_form = st.sidebar.slider("P1 Win % (Last 10)", 0.0, 1.0, 0.7)
p2_form = st.sidebar.slider("P2 Win % (Last 10)", 0.0, 1.0, 0.5)

# Prediction Logic
if st.sidebar.button("Generate Forecast"):
    input_data = pd.DataFrame([[p1_rank, p2_rank, p1_fatigue, p2_fatigue, p1_form, p2_form]], columns=X.columns)
    prediction = model.predict_proba(input_data)[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Player 1 Win Probability", value=f"{prediction[1]:.1%}")
    with col2:
        st.metric(label="Player 2 Win Probability", value=f"{prediction[0]:.1%}")
        
    st.subheader("Feature Importance")
    importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    st.bar_chart(importance.set_index('Feature'))