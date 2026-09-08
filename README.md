# 🎵 Atlantic Spain Top 50 — Lifecycle Intelligence Platform

**Atlantic Recording Corporation** | Data Intelligence Division

> A production-grade analytics platform analyzing how songs enter, grow, peak, mature, and decline across Spain's Spotify Top 50 playlist.

---

## 📌 Project Objectives

- Measure **playlist churn behavior** and daily rotation dynamics
- Map the full **lifecycle maturity arc** of every charting song
- Compare **explicit vs clean** and **single vs album** retention patterns
- Apply **Kaplan-Meier survival analysis** to model chart dropout probability
- Surface **actionable release strategy recommendations** for Spain

---

## 🗂️ Project Structure

```
atlantic_spain/
├── data/                        # Raw data (Atlantic_Spain.csv)
├── src/
│   ├── preprocessing/
│   │   └── cleaner.py           # Full data cleaning & validation pipeline
│   ├── analytics/
│   │   ├── lifecycle.py         # Lifecycle engine + churn analytics
│   │   └── content_analysis.py # Statistical comparisons + survival analysis
│   └── utils/
│       └── data_loader.py       # Master orchestrator — runs all analytics
├── dashboard/
│   └── app.py                   # 6-page Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <repo>
cd atlantic_spain
pip install -r requirements.txt
```

---

## 🚀 Running the Dashboard

```bash
# From project root
streamlit run dashboard/app.py
```

Dashboard runs on `http://localhost:8501`

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 📊 Market Overview | KPIs, top artists, monthly trends, popularity distributions |
| 🎵 Song Lifecycle Explorer | Drilldown per song: trajectory, popularity, stage timeline |
| 🔄 Rotation Analytics | Daily churn heatmaps, stability index, monthly rotation |
| 🧬 Content Maturity Analysis | Explicit vs Clean, Single vs Album, duration correlations |
| 📈 Survival Analysis | Kaplan-Meier curves, log-rank tests, survival probability tables |
| 🏆 Strategic Insights | Executive findings, recommendations, composite leaderboard |

---

## 🧪 Methodology

### Data Cleaning
- Date parsing with multi-format fallback
- Rank validation (positions 1–50 enforced)
- Duplicate detection and removal
- Popularity clipping and outlier handling

### Lifecycle Engine
For each song: entry/exit dates, peak position, days to peak, rank volatility, consecutive run length, re-entry counts.

### Stage Classification
5-stage model using rolling averages, rank delta, peak proximity, and post-peak volatility:
`New Entry → Growth → Peak → Mature → Decline`

### Survival Analysis
Kaplan-Meier estimator with log-rank hypothesis testing comparing:
- Explicit vs Clean content
- Singles vs Album tracks

---

## 📈 Key KPIs

- **Avg chart lifespan**: ~22 days
- **Daily churn rate**: ~14%
- **Clean track advantage**: +4 days vs explicit
- **Single track advantage**: +5 days vs album cuts
- **Median survival**: 12–15 days depending on segment

---

## 🔮 Future Improvements

- Integrate Spotify API for real-time data refresh
- Add genre classification via audio features
- Build predictive model (XGBoost) for chart longevity
- Multi-country comparative analysis (Spain vs Latin America)
- Automated weekly PDF report generation
