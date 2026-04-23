# 🏠 Airbnb Price Prediction

> End-to-end machine learning pipeline predicting Airbnb listing prices — from raw data to SHAP interpretation.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
![Optuna](https://img.shields.io/badge/Optuna-3.4-purple)
![SHAP](https://img.shields.io/badge/SHAP-0.44-red)

---

## 🎯 Objective

Predict the log-price of Airbnb listings from tabular features (location, room type, amenities, host info, reviews) using a complete ML pipeline.

Project completed as part of the **ESILV A3 Machine Learning course** — predictive modeling on a real-world tabular dataset.

---

## 📊 Results

| Model | R² train | R² validation |
|---|---|---|
| LinearSVR (baseline) | 0.37 | 0.36 |
| Random Forest | ~0.76 | ~0.68 |
| LightGBM | 0.78 | 0.71 |
| **LightGBM + Optuna (final)** | **0.785** | **0.715** |

**Final R² = 0.715** on an 80/20 validation split — a **+130% improvement over the 3-feature baseline (0.31)**.

---

## 🛠️ Stack

- **Data**: `pandas`, `NumPy`
- **ML pipeline**: `scikit-learn` (Pipeline, ColumnTransformer, TargetEncoder, StandardScaler, SimpleImputer)
- **Models**: `LinearSVR`, `RandomForestRegressor`, `LightGBM`
- **Hyperparameter tuning**: `Optuna` (Bayesian optimization with persistent SQLite study)
- **Interpretability**: `SHAP`
- **Visualization**: `matplotlib`, `seaborn`

---

## 🔬 Approach

### 1. Feature engineering
- Custom transformer extracting **70+ binary features** from the `amenities` text field (kept only those with ≥1% frequency)
- Date feature: days since reference date (2017-10-05)
- Target encoding for high-cardinality categoricals (neighbourhood, property_type)
- Standard scaling for numerical features
- Missing value imputation via `SimpleImputer`

### 2. Model comparison
Three model families evaluated under the same 5-fold cross-validation protocol:
- **LinearSVR** — baseline, fast but underfits (R² val ≈ 0.36)
- **Random Forest** — captures non-linearities (R² val ≈ 0.68)
- **LightGBM** — gradient boosting, best out-of-the-box performance (R² val ≈ 0.71)

### 3. Hyperparameter tuning
**Bayesian optimization with Optuna** on the LightGBM pipeline:
- Search space: `num_leaves`, `learning_rate`, `max_depth`, `min_child_samples`, `reg_alpha`, `reg_lambda`
- Persistent SQLite study (`optuna_airbnb.db`) — trials accumulate across sessions
- Final gain: **+0.003 R²** vs. untuned LightGBM — marginal, confirming the model was near-saturated

### 4. Interpretability
**SHAP values** computed on the tuned LightGBM → beeswarm plot identifying the top-20 price drivers. Location, number of bedrooms, and review scores emerge as the dominant features.

---

## 📂 Project structure

```
airbnb-price-prediction/
├── Airbnb/
│   └── Explication.txt            # Dataset description (CSVs excluded from git — too large)
├── airbnb_prediction.ipynb        # Main analysis notebook
├── example.ipynb                  # Reference baseline notebook
├── .gitignore
└── README.md
```

> Data files (`airbnb_train.csv`, `airbnb_test.csv`) and the Optuna SQLite study are excluded from git due to size.

---

## ⚙️ Reproduce the results

```bash
git clone https://github.com/Matissegeoffray/airbnb-price-prediction.git
cd airbnb-price-prediction
# Add your own airbnb_train.csv and airbnb_test.csv in Airbnb/
jupyter notebook airbnb_prediction.ipynb
```

---

## 💡 Key takeaways

- **Feature engineering beats model choice**: going from 3 raw features (R² = 0.31) to 70+ engineered features (R² = 0.71) produced a bigger jump than any model swap.
- **Optuna's diminishing returns**: on a well-tuned LightGBM, Bayesian optimization added only +0.003 R². A useful rigor check, but the real wins were upstream in the preprocessing.
- **SHAP confirms domain intuition**: location, bedrooms, and review scores dominate price formation — the model behaves like we'd expect a pricing expert to reason.

---

## 👤 Author

**Matisse Geoffray** — 3rd-year Engineering Student at ESILV

🔍 Looking for a **2-month ML / Data Science internship** (June–July 2026) — Paris or Singapore

📧 matisse.geoffray@gmail.com · [GitHub](https://github.com/Matissegeoffray)
