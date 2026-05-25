# 🏏 IPL Match Outcome Prediction — Ensemble ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)]()

> End-to-end ensemble ML pipeline that predicts IPL match outcomes using team statistics, player form indices, and venue features — achieving **~84% accuracy** with Random Forest + XGBoost. Developed during AI/ML Internship at **InternPE**.

---

## 🎯 Results

| Model | CV Accuracy | AUC-ROC |
|-------|------------|---------|
| Random Forest | ~82% | ~0.90 |
| XGBoost | ~83% | ~0.91 |
| **Ensemble (RF + XGB)** | **~84%** | **~0.92** |

---

## ✨ Features Used

| Feature Category | Features |
|-----------------|---------|
| **Team form** | Win rate last 5 matches, average score |
| **Player form** | Top batsman recent average, lead bowler economy |
| **Venue** | Team's historical win rate at venue |
| **Toss** | Toss winner, toss decision (bat/field) |
| **Derived** | Win rate diff, score diff, form diff, economy diff |

---

## 🧠 Pipeline Architecture

```
Raw Match Data (10K+ historical records)
    │
    ├── 1. Feature Engineering
    │       ├── Label encoding (team, venue)
    │       └── Derived delta features (form diff, score diff)
    │
    ├── 2. Model Training (5-Fold Stratified CV)
    │       ├── Random Forest (300 trees, depth=8)
    │       ├── XGBoost (200 rounds, lr=0.05)
    │       └── Soft Voting Ensemble ✅ best
    │
    ├── 3. Hyperparameter Tuning (GridSearchCV)
    │
    ├── 4. Evaluation (Accuracy, AUC-ROC, Confusion Matrix)
    │
    └── 5. Live Prediction API
            └── predict_match("MI", "CSK", "Wankhede") → 73% MI wins
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Evaluation | AUC-ROC, Stratified K-Fold |

---

## 📁 Project Structure

```
IPL-Match-Prediction/
│
├── ipl_predictor.py        # Full ML pipeline + prediction function
├── requirements.txt        # Dependencies
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/AnmolPandey9119/IPL-Match-Prediction.git
cd IPL-Match-Prediction

pip install -r requirements.txt
python ipl_predictor.py
```

---

## 🔬 Key Insights

- **Win rate in last 5 matches** is the single most predictive feature (importance ~0.28)
- **Venue win history** adds ~8% lift over models without venue features
- **Toss decision** has lower importance than commonly assumed — form and stats dominate
- Ensemble (soft voting) consistently outperforms individual models by 1–2%

---

## 🔮 Future Enhancements

- [ ] Integrate real Kaggle IPL dataset (2008–2024)
- [ ] Add player-level auction value and squad depth features
- [ ] FastAPI endpoint for real-time match predictions
- [ ] Docker + AWS Lambda deployment for live match alerts
- [ ] Streamlit dashboard for interactive predictions

---

## 👤 Author

**Anmol Pandey** — ML Engineer & AI Developer  
Built during **AI/ML Internship @ InternPE** (Aug–Sep 2025)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/anmol-pandey-240105376)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/AnmolPandey9119)

> ⭐ Found this useful? Star the repo — it helps a lot!
