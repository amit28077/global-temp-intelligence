# 🌍 Global Temperature Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-87%25_Accuracy-FF6600?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-8-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end AI-powered climate intelligence platform analyzing 146 years of real NASA data across 20 countries — with future predictions to 2060, multilingual support, and automated policy insights.**

[🌐 Live Demo](#) · [📊 API Docs](#fastapi-endpoints) · [📖 How It Works](#how-it-works)

</div>

---

## 🔥 The Problem This Solves

Every day, millions of people ask:
- **How hot is Earth getting — and how fast?**
- **Which countries are most responsible?**
- **Will we breach the Paris 1.5°C limit — when?**
- **Does switching to renewables actually help?**

This project answers all of it — with **146 years of verified NASA data**, not opinions.

---

## ⭐ What Makes This Different

| Regular DS Projects | This Project |
|---|---|
| Kaggle toy datasets | Real NASA GISS + OWID published data |
| Jupyter notebook only | Full stack: Notebook + FastAPI + Streamlit |
| Single ML model | 3 models: XGBoost + KMeans + Polynomial Regression |
| English only | **8 languages** (Hindi, Marathi, Spanish, French, German, Japanese, Urdu) |
| Past analysis only | **Future predictions up to 2060** |
| No API | **FastAPI with 6 REST endpoints + auto docs** |
| Generic insights | **Rule-based AI engine** — dynamic analysis per country |

---

## 📊 Key Findings From Real Data

```
🌡️  2024: +1.28°C above baseline     →  Hottest year in 146 years (NASA GISS)
🏭  CO2: 422.9 ppm                   →  Highest in 3 million years (NOAA)
🔗  CO2 ↔ Temperature correlation    →  0.97 — almost perfect (proven!)
🇸🇦  Saudi Arabia CO2/person          →  20.4 tonnes (world's worst)
🇳🇴  Norway renewable energy          →  99% (world's best)
🇮🇳  India renewable 2000 → 2024      →  17.5% → 43.2% (fastest improver)
✅  High renewable countries          →  Emit 62% less CO2 than fossil ones
🔮  Paris 1.5°C breach predicted     →  ~2033 if current trend continues
```

---

## 🗂️ Project Structure

```
global-temp-intelligence/
│
├── 📁 data/
│   ├── global_temperature_REAL.csv      ← NASA GISS (1880–2025, 146 years)
│   └── country_climate_REAL.csv         ← OWID + IEA (20 countries, 2000–2024)
│
├── 📁 notebooks/
│   └── Global_Temperature_Intelligence.ipynb   ← Full analysis (12 steps)
│
├── 📁 src/
│   ├── app.py                           ← Streamlit app (6 tabs, 8 languages)
│   ├── api.py                           ← FastAPI backend (6 endpoints)
│   └── translations.py                  ← 8 language translations
│
├── 📁 models/
│   ├── climate_risk_model.pkl           ← Saved XGBoost model
│   ├── scaler.pkl                       ← StandardScaler
│   └── feature_names.pkl               ← Feature column names
│
├── 📁 reports/                          ← 11 generated chart images
│
├── requirements.txt
└── README.md
```

---

## 🔬 How It Works

```
REAL DATA (NASA GISS + OWID)
         ↓
    EDA — 11 Charts
    Anomaly Detection (Z-Score)
    Correlation Analysis (r = 0.97)
         ↓
  FEATURE ENGINEERING
  → Fossil_Dependency = 100 - Renewable%
  → Carbon_Efficiency = CO2 / (Fossil + 1)
  → Vulnerability_Index (composite)
  → Crisis_Level (target: 0/1/2)
         ↓
     ML MODELS
  ┌─────────────────────────────┐
  │ XGBoost       → 87% accuracy│
  │ KMeans        → 3 clusters  │
  │ Poly Regression → 2060 forecast│
  └─────────────────────────────┘
         ↓
  SHAP EXPLAINABILITY
  CO2/capita = #1 risk driver
         ↓
     DEPLOYMENT
  ┌────────────────────────────────────┐
  │ FastAPI  (6 REST endpoints)        │
  │ Streamlit (6 tabs, 8 languages)    │
  │ AI Insights Engine (rule-based ML) │
  └────────────────────────────────────┘
```

---

## 🤖 ML Models

### XGBoost Classifier — Climate Risk Prediction

**Input Features (8):**
- CO2 per capita, Renewable energy %, Temperature anomaly
- Extreme weather events, Forest cover %
- Fossil dependency, Carbon efficiency, Vulnerability index

**Results:**
```
               precision  recall  f1-score
STABLE              0.90    0.85      0.88
AT RISK             0.79    0.82      0.81
CRITICAL            0.91    0.94      0.93

Overall Accuracy:  87%
Cross Validation:  5-fold → 87% ± 2%
```

### KMeans Clustering (K=3, Elbow Method)

| Cluster | Countries | Avg CO2 | Avg Renewable |
|---------|-----------|---------|---------------|
| 🟢 Climate Leaders | Norway, Sweden, Brazil | 4.2t | 78% |
| 🟡 In Transition | India, Germany, UK, France | 5.1t | 41% |
| 🔴 Climate Laggards | Saudi Arabia, USA, Australia, Russia | 15.2t | 20% |

### Polynomial Regression — Temperature Forecast

```
2025: +1.01°C  |  2030: +1.13°C  |  2035: +1.25°C
2040: +1.37°C  |  2045: +1.50°C  |  2050: +1.63°C  ← DANGER zone
```

---

## ⚡ FastAPI Endpoints

```
Base URL: http://localhost:8000
Docs:     http://localhost:8000/docs   (Swagger UI — auto generated)

GET  /              → API info + all endpoint descriptions
GET  /health        → Model loading status + accuracy
GET  /global-stats  → Key global climate statistics
GET  /countries     → Profiles for all 20 countries
POST /predict       → Predict climate risk for any input
POST /ai-report     → Generate data-driven climate insights
```

**Example POST /predict:**
```json
{
  "country": "India",
  "co2_per_capita": 2.2,
  "renewable_pct": 43.2,
  "temp_anomaly": 1.28,
  "extreme_events": 46,
  "forest_cover_pct": 24.8
}
```

**Response:**
```json
{
  "country": "India",
  "risk_level": "🟢 STABLE",
  "risk_score": 26.7,
  "top_risk_factor": "Fossil dependency",
  "recommendation": "Maintain trajectory. Increase renewable to 60% by 2030."
}
```

---

## 🌐 Streamlit Dashboard — 6 Tabs

| Tab | What You See |
|-----|--------------|
| 🌡️ 146 Years Data | NASA temperature bar chart, decade analysis, correlation heatmap |
| 🌐 Country Analysis | Year slider, CO2/renewable comparison, single country deep dive |
| 🔮 Future Predictions | Temperature forecast 2060, country CO2/renewable forecasts, 2030 rankings |
| 🤖 ML Risk Model | Interactive sliders, real-time prediction, gauge chart, vs real countries |
| 🌱 Solutions | Renewable vs fossil proof, country cards, trend lines |
| 📊 AI Insights | Country select → 3-section analysis (situation, causes, actions) |

### 🗣️ 8 Language Support
English · हिंदी Hindi · मराठी Marathi · Español · Français · Deutsche · 日本語 · اردو

---

## 📡 Data Sources

| Source | Data | Years | Credibility |
|--------|------|-------|-------------|
| NASA GISS | Global temperature anomaly | 1880–2025 | America's Space Agency |
| Our World in Data | CO2, energy, population | 2000–2024 | Oxford University backed |
| IEA | Renewable energy % | 2000–2024 | International Energy Agency |
| NOAA | CO2 concentration | 1959–2024 | US Atmospheric Scientists |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/global-temp-intelligence.git
cd global-temp-intelligence

# 2. Install
pip install -r requirements.txt

# 3. Launch Streamlit App
streamlit run src/app.py

# 4. Launch FastAPI (new terminal)
uvicorn src.api:app --reload
# Visit: http://localhost:8000/docs
```

---

## 🧪 Skills Demonstrated

```
Data Science:
✅ Real data integration (NASA + OWID — not synthetic)
✅ EDA with 11 professional visualizations
✅ Statistical analysis (Z-score, Pearson correlation r=0.97)
✅ Feature engineering (5 new features created)

Machine Learning:
✅ Supervised learning — XGBoost (87% accuracy, 5-fold CV)
✅ Unsupervised learning — KMeans clustering
✅ Time series prediction — Polynomial Regression (2060 forecast)
✅ SHAP explainability (bar + beeswarm plots)
✅ Full model evaluation (F1, confusion matrix, CV)

Engineering:
✅ REST API — FastAPI (6 endpoints, auto Swagger docs)
✅ Interactive dashboard — Streamlit (6 tabs)
✅ Rule-based AI insights engine
✅ Multilingual application (8 languages)
✅ Production-ready code structure
```

---

## 👤 About

**Amit Vilas More**
B.Tech in AI and Data Science | Sharad Institute of Technology, Kolhapur | Pune

📧 moreamit7887@gmail.com · 📱 +91 7887962012

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/YOUR_USERNAME)

---

## 📄 License

MIT License — Data from NASA GISS (public domain) and Our World in Data (CC BY).

---

<div align="center">

⭐ **Star this repo if it helped you!**

*Built with real data. Powered by ML. Made for impact.*

**Guided by CampusX — 100 Days of Machine Learning**

</div>
