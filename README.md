# 🎾 ATP Fatigue Forecaster
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://atp-fatigue-forecaster-amgk86xppbhgdmajt5txdm.streamlit.app/)
[![CI Build](https://github.com/vishaan2010-dotcom/atp-fatigue-forecaster/actions/workflows/python-tests.yml/badge.svg)](https://github.com/vishaan2010-dotcom/atp-fatigue-forecaster/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Data Source](https://img.shields.io/badge/Data-JeffSackmann%2Ftennis__atp-green)](https://github.com/JeffSackmann/tennis_atp)

A production-grade machine learning application that forecasts professional tennis match outcomes by quantifying physiological and cognitive fatigue. 

<img width="1824" height="892" alt="image" src="https://github.com/user-attachments/assets/d3dc182c-c678-4f1c-b1b1-d11fba44ac18" />


## 📖 The Science: NHSJS Research Context
Traditional tennis analytics over-index on baseline Elo ratings or ATP rankings. However, in high-stakes environments (e.g., USTA Level 2 Nationals or ATP Masters 1000s), cumulative physical attrition alters baseline win probabilities. 

This project mathematically applies the findings from my systematic review: ***THE BREAKING POINT: A SYSTEMATIC REVIEW OF PHYSIOLOGICAL AND COGNITIVE FATIGUE EFFECTS ON PROFESSIONAL TENNIS PERFORMANCE***. The model engineers a synthetic "Cumulative Fatigue" feature derived from historical match duration to override standard rank-based predictions when a player reaches a physiological breaking point.

## ⚙️ Technical Architecture
* **Frontend UI:** Streamlit (Hosted live with real-time inference)
* **Machine Learning:** Scikit-Learn (Random Forest Classifier, `n_estimators=100`)
* **Data Pipeline:** Pandas & NumPy (Vectorized operations fetching live 2023 ATP data)
* **CI/CD Automation:** GitHub Actions (Automated PyTest validation on every commit)

## 🧠 Feature Engineering
The model ingests real match data and synthesizes constraints that a standard ATP ranking misses:
1. **Cumulative Minutes:** Calculates physical load based on recent match times.
2. **Form / Momentum:** Recent win percentages over a 10-match rolling window.
3. **Rank Differential:** Baseline comparative skill level.

## 🚀 Quick Start (Run Locally)
Want to run the model on your own machine? 

```bash
# Clone the repository
git clone [https://github.com/vishaan2010-dotcom/atp-fatigue-forecaster.git](https://github.com/vishaan2010-dotcom/atp-fatigue-forecaster.git)

# Navigate into the directory
cd atp-fatigue-forecaster

# Install the required dependencies
pip install -r requirements.txt

# Launch the Streamlit server
streamlit run src/app.py
