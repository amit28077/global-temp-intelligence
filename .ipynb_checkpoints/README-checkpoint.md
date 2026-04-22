# 🌍 Global Temperature Intelligence System

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://streamlit.io)
[![Claude AI](https://img.shields.io/badge/Claude-AI%20Powered-purple)](https://anthropic.com)

> **140 years of real climate data · ML risk classification · Claude AI policy reports · Interactive global dashboard**

---

## 🔥 What Makes This Different

Every climate project shows you charts. This project **thinks** about the data.

- Analyzes **140 years** of NASA + NOAA temperature records
- Detects **climate anomaly years** statistically (Z-score method)
- **Clusters 20 countries** into Climate Leaders / In Transition / Laggards
- **ML model classifies** climate risk: Stable / At Risk / Critical
- **Claude AI writes** a UN-level policy report for any country
- **Interactive Streamlit app** — real-time risk simulator

---

## 📊 The 5 Core Questions This Project Answers

| Question | Method |
|---------|--------|
| How much has Earth warmed? | 140-year EDA |
| Which decade was the worst? | Decade-by-decade analysis |
| Which countries are most responsible? | Country CO2 + renewable comparison |
| What drives climate risk the most? | XGBoost Feature Importance |
| Does going green actually work? | Renewable vs CO2 correlation proof |

---

## 🗂️ Project Structure

```
global-temp-intelligence/
│
├── 📁 data/
│   ├── global_temperature.csv      ← 145 years, 7 climate variables
│   ├── country_climate.csv         ← 20 countries × 25 years × 8 features
│   └── generate_data.py            ← Data generation script
│
├── 📁 notebooks/
│   └── Global_Temperature_Intelligence.ipynb  ← Full analysis (12 steps)
│
├── 📁 src/
│   └── app.py                      ← Streamlit web app (5 tabs)
│
├── 📁 models/
│   ├── climate_risk_model.pkl      ← Saved XGBoost model
│   └── scaler.pkl                  ← Saved StandardScaler
│
├── 📁 reports/
│   ├── 01_global_overview.png
│   ├── 02_decade_analysis.png
│   ├── 03_country_analysis.png
│   ├── 04_anomaly_detection.png
│   ├── 05_clustering.png
│   ├── 06_confusion_matrix.png
│   ├── 07_feature_importance.png
│   └── 08_solution_analysis.png
│
├── requirements.txt
└── README.md
```

---

## 🧠 CampusX Skills Used — Every Single One

| CampusX Topic | Applied Here |
|--------------|-------------|
| Python + Pandas | Merge 3 datasets, 145 years of data |
| EDA + Matplotlib/Seaborn | 8 professional visualizations |
| Statistical Analysis | Z-score anomaly detection |
| Feature Engineering | 5 new climate features created |
| Correlation Analysis | CO2 ↔ Temperature = 0.97 correlation |
| KMeans Clustering | Country grouping: Leaders/Laggards |
| Random Forest | Climate risk classification |
| XGBoost | Best performing model |
| Cross Validation | 5-fold CV for all models |
| Feature Importance | What drives climate risk most |
| Model Saving | joblib pkl files |
| LLM Integration | Claude API — UN-style policy reports |
| Streamlit Deployment | 5-tab interactive dashboard |

---

## 🌡️ Key Findings From The Data

```
🌡️  Earth is 1.4°C hotter than in 1880
🏭  CO2 has risen from 280 → 421 ppm (50% increase)
🌊  Sea level rose 250mm in 124 years
🧊  Arctic lost 4 million km² of ice
📅  2020s are the HOTTEST decade in recorded history
🇸🇦  Saudi Arabia emits 16.8 tonnes CO2 per person/year
🇳🇴  Norway runs on 98% renewable energy
🇮🇳  India: moderate CO2 but high vulnerability to climate impact
✅  Countries with >50% renewable emit 60% less CO2
```

---

## 🚀 How to Run

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/global-temp-intelligence.git
cd global-temp-intelligence

# 2. Install
pip install -r requirements.txt

# 3. Generate data
python data/generate_data.py

# 4. Run notebook
jupyter notebook notebooks/Global_Temperature_Intelligence.ipynb

# 5. Launch app
streamlit run src/app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| Pandas + NumPy | Data processing |
| Matplotlib + Seaborn | Static visualizations |
| Plotly | Interactive charts |
| Scikit-Learn | KMeans, Random Forest, preprocessing |
| XGBoost | Best ML model |
| Claude API | AI policy report generation |
| Streamlit | Web application |
| Joblib | Model persistence |

---

## 📡 Data Sources

| Source | Data |
|--------|------|
| NASA GISS | Global surface temperature records |
| NOAA | Atmospheric CO2 measurements |
| National Snow and Ice Data Center | Arctic ice extent |
| IEA (International Energy Agency) | Energy mix per country |
| World Bank | Country-level climate indicators |
| Our World in Data | CO2 per capita, renewable % |

---

## 💼 What This Project Demonstrates to Recruiters

1. **End-to-end thinking** — from raw data to deployed app
2. **Real-world problem** — climate change is the #1 global challenge
3. **Multiple ML techniques** — clustering + classification + anomaly detection
4. **LLM integration** — not just ML, but AI-powered insights
5. **Data storytelling** — 8 visualizations that tell a compelling story
6. **Deployment** — working Streamlit app, not just a notebook

---

## 👤 About

**Amit More** | Data Science Enthusiast | Python · ML · Climate Tech

📧 moreamit7887@gmail.com  


---

*Guided by: CampusX YouTube — 100 Days of ML + ML Algorithms + DSMP*

⭐ **Star this repo if it helped you!**
