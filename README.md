# ATP Fatigue Forecaster
### An Interactive ML Dashboard for Physiological Load Modeling in Professional Tennis

> **Live Demo →** https://atp-fatigue-forecaster-amgk86xppbhgdmajt5txdm.streamlit.app/

Built as a direct computational extension of my systematic review:
*"The Breaking Point: A Systematic Review of Physiological and Cognitive Fatigue Effects on Professional Tennis Performance"* (2025)

---

## The Research Question

Traditional ATP forecasting relies almost entirely on static ranking systems like Elo. But rankings don't know if a player spent 800 minutes on court in the last two weeks. They don't know if he played five matches in eight days and flew across time zones between them.

This project tests a simple hypothesis: **physiological load accumulation — computed from real match scheduling data — adds meaningful predictive signal beyond ranking alone.**

---

## The Precision Degradation Cascade

The systematic review (PRISMA guidelines, 847 records screened, 10 studies included) identified a consistent 5-stage sequence of performance breakdown in elite tennis players under fatigue:

| Stage | Name | Key Finding |
|-------|------|-------------|
| 1 | Lower-Body Fatigue | Knee flexion reduces 15–23°; ground reaction force drops |
| 2 | Kinetic Chain Compensation | Trunk rotation increases 8–12% to preserve ball speed |
| 3 | Precision Loss | Serve accuracy −25–32%; groundstroke accuracy up to −69% |
| 4 | Range of Motion Restriction | Hip rotation ROM −13°; shots flatten and land long |
| 5 | Cognitive Failure | Reaction time +47–68 ms; decision-making quality −18–34% |

The counterintuitive core finding: **serve velocity declined only 0.4–3.1%** while accuracy collapsed. Fatigue doesn't slow players down — it makes them imprecise. The model's features are designed to operationalize Stage 1 load accumulation as an early warning signal for downstream cascade effects.

---

## Features

- **Player Lookup Mode** — select any ATP player by name and auto-populate their real rolling stats from historical records
- **Real Rolling Features** — cumulative court minutes in 7/14/28-day windows, rest days, rolling win rates, tournament load — all computed from actual ATP scheduling data, no synthetic proxies
- **Calibrated Gradient Boosting** — isotonic regression calibration ensures probability outputs are statistically reliable
- **Cascade-Linked Commentary** — inference output explicitly maps predictions back to the 5-stage research model
- **Key Findings Tab** — interactive visualization of the systematic review's core results
- **Reliability Diagram** — model calibration chart to verify probability accuracy

---

## How It Works

```
Raw ATP Match Data (JeffSackmann/tennis_atp)
        │
        ▼
Rolling Feature Engineering
  ├── Cumulative court minutes (7d / 14d / 28d windows)
  ├── Days since last match
  ├── Matches played in prior 7 days
  ├── Rolling win % (last 10 and 20 matches)
  └── Tournament-level load (matches + minutes in current draw)
        │
        ▼
Calibrated Gradient Boosting Classifier
  ├── 300 estimators, learning rate 0.05, max depth 4
  ├── Isotonic regression probability calibration (CalibratedClassifierCV)
  └── Temporal train/test split — train on older seasons, test on newer ones
        │
        ▼
Match Outcome Probability + Cascade Stage Analysis
```

**Critical design decision:** Features are computed using only data available *before* each match. The history is updated *after* features are extracted, preventing any lookahead bias.

---

## Data Source

[JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) — the most comprehensive public ATP match-level dataset available, covering match results, rankings, scores, and match duration from 1968 to present. This project fetches the 4 most recent available years dynamically at runtime.

---

## Running Locally

```bash
git clone https://github.com/vishaan2010-dotcom/atp-fatigue-forecaster
cd atp-fatigue-forecaster
pip install -r requirements.txt
streamlit run app.py


Requires Python 3.9+. First run will take 2–3 minutes to fetch data and train the model; subsequent runs use Streamlit's cache.

---

## Research Foundation

**Title:** The Breaking Point: A Systematic Review of Physiological and Cognitive Fatigue Effects on Professional Tennis Performance

**Methods:** PRISMA-compliant systematic review. Boolean search across PubMed, Google Scholar, and SPORTDiscus (January 2002 – December 2023). 847 records screened → 10 studies meeting inclusion criteria (elite/sub-elite populations, validated fatigue protocols, quantitative outcome measures).

**Key Sources:** Davey et al. (2002), Hornery et al. (2007), Girard et al. (2008), Lyons et al. (2013), Reid & Duffield (2014), Bilić et al. (2023)

---

## Limitations

- Match duration (minutes) is the best available proxy for physiological load in public ATP data. Direct biomechanical measurements (knee flexion, EMG) are not publicly recorded at scale.
- The model captures scheduling load but cannot directly observe Stage 1–3 cascade variables. These remain latent signals approximated by court time accumulation.
- Heterogeneity in match conditions (altitude, heat, indoor vs. outdoor) is not controlled for, mirroring the measurement heterogeneity that prevented formal meta-analysis in the systematic review.

---

*This project is for academic and research purposes only.*