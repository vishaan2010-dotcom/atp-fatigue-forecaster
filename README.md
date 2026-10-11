# ATP Fatigue Forecaster

An interactive Streamlit ML app for studying physiological load, fatigue, and match-outcome forecasting in professional men's tennis.

**Live demo:** https://atp-fatigue-forecaster-amgk86xppbhgdmajt5txdm.streamlit.app/

This project extends my published systematic review, "The Breaking Point: A Systematic Review of Physiological and Cognitive Fatigue Effects on Professional Tennis Performance," published in the National High School Journal of Science in 2026. DOI: https://doi.org/10.5281/zenodo.21271049

## Research Motivation

Traditional tennis forecasting leans heavily on rankings and rating systems. Those signals are useful, but they do not directly represent recent court time, short rest, tournament load, or surface-specific physical demand.

The ATP Fatigue Forecaster tests whether scheduling-derived load features add predictive information beyond ranking alone. The app operationalizes fatigue concepts from the review as pre-match features, then compares models and ablations on held-out historical ATP match data.

## Research Findings Used In The App

The app only uses the following quantitative findings from the paper:

| Finding | Reported value |
| --- | --- |
| Serve velocity decline under fatigue | 0.4-2.8% |
| Serve accuracy decline | 25-41% |
| Groundstroke accuracy decline | up to 69% |
| Knee flexion reduction | ~6 degrees over 3 sets, Fenter et al. 2017 |
| Reaction time delay | 47-68 ms |
| Decision-making | qualitative declines, no percentage |

These findings motivate the Precision Degradation Cascade shown in the app: fatigue has a larger effect on precision, movement quality, and cognition than on raw serve speed.

## Models

The app contains two separate models.

**Pre-match fatigue model**

This is the main ATP Fatigue Forecaster model. It predicts match outcome from ranking, rolling court-time load, rest, tournament load, recent form, head-to-head history, opponent quality, and surface. It uses a Gradient Boosting classifier with isotonic probability calibration through CalibratedClassifierCV.

Validation uses a chronological positional hold-out on date-sorted data: the first training portion is fit, and the last held-out portion is evaluated. This is intentionally described as a positional hold-out, not as a full production backtest with tournament-level embargoing.

**In-match score-state model**

This secondary model estimates match win probability from score state only: sets, games, server, point number, match format, tiebreak status, and decisive-set state. It uses point-by-point data from the Match Charting Project. It is intentionally uncalibrated because isotonic calibration was tested and kept out after worse Brier and log-loss performance on this dataset.

## Data Integrity and Leakage Prevention

Preventing look-ahead leakage is the most important safeguard in this project, since it is the most common way a sports-prediction model appears strong but is actually invalid. Specific measures:

- Every feature is computed before the current match is added to a player's history, so no rolling window can include the match it is predicting.
- Matches are sorted by date and within-tournament order using a stable sort. A tournament's start date is shared by all of its rounds, so a naive date sort can place a final ahead of a first-round match of the same event and leak later-round results into prior history. The stable, order-aware sort prevents this.
- Player labels (P1 and P2) are randomized so the target is not tied to winner/loser ordering.
- The in-match model is split by match, not by point, so points from one match never appear in both training and test.
- A built-in leak-check diagnostic reports mean feature values by outcome. If the features are clean, winners and losers should look nearly identical before the match begins.

## Ablation Study

The Ablation Study tab tests the central research question directly. It trains comparable Gradient Boosting models with progressively richer feature groups:

1. Ranking only
2. Ranking plus physiological load
3. Load plus recent form
4. Head-to-head and opponent quality
5. Full feature set

The key comparison is ranking-only versus ranking plus physiological load. The app reports the AUC lift dynamically from the current fetched dataset, showing whether scheduling-derived load contributes separable predictive signal beyond rank.

### Ablation and Significance

The research hypothesis is that accumulated physiological load adds predictive
signal beyond static ranking. A single-run AUC cannot test that, because the
gradient booster subsamples on every fit. `src/stability_test.py` repeats each
feature tier across 21 model seeds on an identical chronological hold-out, then
runs a 2,000-sample paired bootstrap over the test matches. Predictions are
averaged across seeds before bootstrapping, so the interval reflects sampling
uncertainty rather than the luck of one seed, and the single-seed result is
printed alongside as a sensitivity check. The P1/P2 label randomisation in
`engineer_features` is explicitly seeded; without that the dataset itself differs
between runs and the entire table shifts.

Data seed 42, 17,685 matches, 14,148 train / 3,537 test.

| Tier | Features | Test AUC |
|---|---|---|
| Ranking only | 2 | 0.6809 +/- 0.0007 |
| + Physiological load | 20 | 0.6873 +/- 0.0012 |
| + Form | 24 | 0.6930 +/- 0.0011 |
| + H2H + opponent quality | 28 | 0.6932 +/- 0.0012 |
| Full | 29 | 0.6932 +/- 0.0012 |

Every adjacent step is tested, so no tier is dismissed on a comparison that was
never run:

| Comparison | Delta AUC | 95% CI | Verdict |
|---|---|---|---|
| Load vs ranking | +0.0073 | [+0.0008, +0.0136] | excludes zero |
| Form vs load | +0.0057 | [+0.0015, +0.0099] | excludes zero (seed-averaged only) |
| H2H + quality vs form | +0.0003 | [-0.0024, +0.0031] | crosses zero |
| Full vs H2H + quality | +0.0000 | [-0.0004, +0.0005] | crosses zero |
| Full vs ranking | +0.0132 | [+0.0051, +0.0209] | excludes zero |

**The hypothesis is supported.** Physiological load improves AUC over ranking
alone, positive in 21 of 21 seeds, and the confidence interval excluded zero in
all four analyses run during development: two datasets, each under the
seed-averaged and single-seed bootstrap. The effect is small. The interval's
lower bound sits just above zero, so the data is consistent with load being worth
a very small fraction of a percentage point of AUC. It is detectable, not
substantial.

Recent form is positive in every seed but was distinguishable from zero in only
one of those four analyses, so it is not reported as an established effect.
Head-to-head record, opponent quality and surface could not be distinguished from
zero in any of them.

Reproducibility: the pinned data source and fixed seeds make the pipeline
deterministic on a given machine. Figures may still differ in the fourth decimal
across machines because of library versions, so the tier ordering and the
zero-crossing verdicts are the findings, not the individual digits.

Data provenance: the upstream `JeffSackmann/tennis_atp` repository no longer
resolves. Match data for 2018 to 2024 is read from a fork of it pinned at commit
`6b9482f`. A cross-check against an independent fork, mneedham/tennis_atp, was
inconclusive: it shares the 49-column schema but is a snapshot of an earlier
upstream revision, differing in every data row for an overlapping season and
missing 2023 and 2024 entirely. Sackmann revised these files over time and no
authoritative copy remains reachable, so the data's fidelity to his released
files cannot be established, only assumed.

## Data Sources

- **Jeff Sackmann tennis_atp:** canonical ATP match results, rankings, dates, surfaces, scores, and durations. Match data is read from a single pinned source: a fork of JeffSackmann/tennis_atp at commit 6b9482f, covering 2018 to 2024. The upstream repository no longer resolves and the previous third-party mirror fallbacks were removed, because silently substituting an unverified source was itself the defect. If the pinned source becomes unavailable the app loads no match data rather than quietly training on something else.
- **Jeff Sackmann Match Charting Project:** point-by-point charting data for the in-match score-state model.

The app fetches public CSV files at runtime and caches them with Streamlit. Network failures, empty responses, and HTTP errors are handled with user-facing messages.

## Honest Limitations

- Public ATP data does not include direct biomechanical measurements such as knee flexion or EMG, so physiological load is approximated from match duration and scheduling history.
- Match duration is an imperfect proxy for physical and cognitive fatigue because it cannot capture heat, travel, illness, playing style, or rally intensity.
- The pre-match split is a chronological positional hold-out on date-sorted data, not a full production backtest with tournament-level embargoing.
- The in-match model does not know player identity, ranking, fatigue, serve quality, rally length, or point-tracking data.
- The app is for academic and research use only. It is not intended for betting or wagering.

## Local Setup

```
git clone https://github.com/vishaan2010-dotcom/atp-fatigue-forecaster
cd atp-fatigue-forecaster
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py
```

On macOS or Linux, activate the environment with:

```
source .venv/bin/activate
```

The first run fetches public CSV data and trains the cached models. Later runs are faster because Streamlit reuses cached data and model objects.

## Project Structure

```
src/app.py                  Streamlit app and model pipeline
src/test_app.py             Data-loading and feature-engineering tests
src/train_inmatch.py        Standalone in-match model training script
src/calibrate_inmatch.py    Calibration comparison script for the in-match model
src/explore_pbp.py          Match Charting Project data exploration helper
requirements.txt            Minimal Streamlit Community Cloud dependencies
```

## License

This project is released for academic and research purposes under the repository license.
