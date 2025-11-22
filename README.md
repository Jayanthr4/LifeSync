# ⚡ LifeSync: Relocation Risk Engine

[![Python Version](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Aligning your budget, health, and vibe for the modern mover.

---

## 📋 Overview

LifeSync is a predictive analytics dashboard designed to help individuals make smarter relocation decisions.

Most moving apps only answer one question: "Can I afford the rent?" LifeSync answers the question: "Can I afford to live here?"

By synthesizing data from Housing, Healthcare, Nutrition, and Entertainment, the engine calculates a personal "Survival Score" (0-100%) for moving to New York City, factoring in hidden liabilities like predicted medical costs and lifestyle stress.

---

## 💡 The Problem vs. The Solution

| The "Rent Trap" (Old Way) | The LifeSync Way (New Way) |
|---------------------------|---------------------------|
| Looks only at Gross Income vs. Rent. | Calculates Real Disposable Income (Income - Health Liability). |
| Ignores chronic health costs. | Uses Machine Learning to predict medical costs based on Age/BMI. |
| Ignores mental stress/vibe. | Matches neighborhood energy to Spotify audio features. |
| Result: Financial Stress. | Result: Calculated Risk Assessment. |

---

## ⚙️ How It Works (The Logic)

LifeSync operates on a unique actuarial logic:

1. **Predict Liability**: Random Forest Regressor trained on demographic health data to predict monthly medical costs based on Age, BMI, and Smoker Status.
2. **Calculate Real Budget**: Total Budget - Predicted Medical Cost = Real Housing Budget.
3. **Market Validation**: Cross-reference Real Housing Budget against NYC Airbnb market data to determine affordability.
4. **Risk Score**: Normalized risk score (0-100%):
   - 0-40%: Stable (Surplus income)  
   - 41-70%: Moderate Risk (Breaking even)  
   - 71-100%: Critical Risk (Projected debt)  

---

## 🚀 Features

1. **Actuarial Risk Engine**: Real-time adjustment of financial risk based on biological factors.  
2. **Geospatial Analysis**: Interactive map of NYC boroughs filtering apartments by your specific remaining budget after health costs.  
3. **Lifestyle Mitigation**:
   - Dietary Intervention: Suggests low-calorie/healthy meal plans based on BMI.  
   - Stress Management: Generates Spotify playlists to counter relocation anxiety.

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+  
- **Frontend:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-Learn (Random Forest Regressor)  
- **Visualization:** Plotly Express  
- **Model Persistence:** Joblib  

---

## 📂 Datasets

This project uses 4 datasets:

1. **Medical Insurance Cost Prediction:** For training the risk model.  
2. **Airbnb Open Data (NYC 2019):** Housing market baseline.  
3. **Spotify Tracks Dataset:** Vibe/genre matching.  
4. **Recipes Dataset (64k Dishes):** Dietary recommendations.

> Note: Place all CSV files in a folder named `Data_Spark_dataset/` in the root directory.

---

## 💻 Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/lifesync.git
cd lifesync
# Install dependencies
pip install streamlit pandas numpy scikit-learn plotly joblib

# Run the application
streamlit run App.py
