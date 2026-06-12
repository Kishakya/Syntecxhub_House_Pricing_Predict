# 🏠 House Price Prediction — Linear Regression

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project builds a **Linear Regression model** to predict median house prices in California based on demographic and geographic features. The goal is to demonstrate a complete machine learning pipeline — from raw data to saved model — including data cleaning, feature engineering, model training, and evaluation.

---

## 📂 Dataset Description

| Property | Details |
|---|---|
| **Source** | California Housing Dataset |
| **Rows** | 20,640 |
| **Columns** | 10 |
| **Target** | `median_house_value` |
| **Missing Values** | `total_bedrooms` — 207 missing (imputed with median) |

### Features

| Feature | Type | Description |
|---|---|---|
| `longitude` | Float | Geographic longitude |
| `latitude` | Float | Geographic latitude |
| `housing_median_age` | Float | Median age of houses in block |
| `total_rooms` | Float | Total rooms in block |
| `total_bedrooms` | Float | Total bedrooms in block |
| `population` | Float | Block population |
| `households` | Float | Number of households |
| `median_income` | Float | Median income (tens of thousands USD) |
| `ocean_proximity` | Categorical | Distance category from ocean |
| `median_house_value` | Float | **Target** — Median house value (USD) |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations, log transformation |
| `matplotlib` & `seaborn` | Data visualization |
| `scikit-learn` | Preprocessing, model training, evaluation |
| `joblib` | Model serialization |

---

## 🤖 Model Used

### Linear Regression (with Polynomial Features)

Plain Linear Regression was used as required, enhanced with **degree-2 interaction terms** via `PolynomialFeatures` to capture non-linear relationships while keeping the model family linear.

**Pipeline:**
1. Impute missing values with median
2. Engineer 6 new features (ratios)
3. One-hot encode `ocean_proximity`
4. Log-transform target (`np.log1p`)
5. Standardize features (`StandardScaler`)
6. Expand features with `PolynomialFeatures(degree=2, interaction_only=True)`
7. Fit `LinearRegression`

**Engineered Features:**

```
rooms_per_household       = total_rooms    / households
bedrooms_per_room         = total_bedrooms / total_rooms
population_per_household  = population     / households
rooms_per_person          = total_rooms    / population
bedrooms_per_person       = total_bedrooms / population
income_per_room           = median_income  / total_rooms
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error — penalizes large errors more heavily |
| **MAE** | Mean Absolute Error — average absolute difference in USD |
| **R²** | Proportion of variance explained by the model (1.0 = perfect) |

> All metrics are computed on the **original dollar scale** after inverting the log transform with `np.expm1`.

---

## 📈 Results

| Metric | Score |
|---|---|
| **RMSE** | $57,526 |
| **MAE** | $38,926 |
| **R²** | 0.6681 |

**Key observations:**
- The log transformation of the target significantly reduced skew and improved model fit
- Polynomial interaction features provided a meaningful boost over plain Linear Regression
- Residual analysis shows the model is well-centered around zero with mild heteroscedasticity at mid-range values — a known limitation of linear models on geographic data

---

## 📁 Project Structure

```
house-price-prediction/
│
├── housing.csv                    # Raw dataset
├── house_price_prediction.ipynb   # Main notebook
├── house_price_model.pkl          # Saved model
├── scaler.pkl                     # Saved scaler
├── poly.pkl                       # Saved polynomial features
└── README.md                      # Project documentation
```

---

## 👤 Author

**Kishkaya Gayathri**
Bachelor of Science in Information Technology specialized in Artificial Intelligence
[SLIIT] — [2026]
