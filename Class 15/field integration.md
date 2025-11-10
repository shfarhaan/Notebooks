# Class 15: EE–ML Integration — Case Studies & Debugging Models

**Lab:** Guided project coding
**Project Update:** code review

---

### Overview

Models aren’t magic boxes — they fail, drift, and break just like circuits. This class connects EE case studies (renewable energy forecasting, fault detection, IoT health monitoring) to ML workflows and shows systematic debugging: **data → model → evaluation → fix cycle**.

---

### Lecture Notes (Key Ideas)

* **Case Studies:**

  * Renewable energy forecasting (solar/wind).
  * Fault detection in transformers/power lines.
  * Predictive maintenance (vibration, thermal).
* **Debugging Levels:**

  1. **Data Debugging:** missing values, leakage, imbalance.
  2. **Model Debugging:** bias/variance, wrong assumptions, wrong features.
  3. **Evaluation Debugging:** wrong metric, test leakage.
* **Debugging Tools:**

  * Learning curves → bias vs variance.
  * Residual analysis → patterns in errors.
  * Feature importance / SHAP → interpretability.
  * Sanity checks → shuffle labels, random baselines.

---

## Demo Notebook (Colab-style)

### 0) Example EE Dataset: Solar power forecasting (synthetic)

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
np.random.seed(15)

days = 365
t = np.arange(days)
# base sinusoidal daily pattern + trend + noise
solar = 5 + 3*np.sin(2*np.pi*t/365*4) + 0.02*t + np.random.normal(0,0.5,days)
# features: day, seasonality, noise
df = pd.DataFrame({
    "day": t,
    "tempC": 20 + 10*np.sin(2*np.pi*t/365*2) + np.random.normal(0,1,days),
    "humidity": 50 + 20*np.sin(2*np.pi*t/365) + np.random.normal(0,5,days),
    "solar_output": solar
})
df.head()
```

### 1) Train/test split and baseline regression

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = df[["day","tempC","humidity"]]
y = df["solar_output"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

rf = RandomForestRegressor(n_estimators=200, random_state=0)
rf.fit(X_train,y_train)
pred = rf.predict(X_test)

rmse = mean_squared_error(y_test,pred,squared=False)
print("Baseline RF RMSE:", rmse)
```

### 2) Debugging — learning curves

```python
from sklearn.model_selection import learning_curve

sizes, train_scores, val_scores = learning_curve(rf, X_train,y_train,
                                                 cv=5, scoring="neg_root_mean_squared_error",
                                                 train_sizes=np.linspace(0.1,1.0,6), n_jobs=-1)

train_rmse = -train_scores.mean(axis=1)
val_rmse   = -val_scores.mean(axis=1)

plt.plot(sizes,train_rmse,"o-",label="Train RMSE")
plt.plot(sizes,val_rmse,"o-",label="CV RMSE")
plt.xlabel("Training Samples"); plt.ylabel("RMSE")
plt.title("Learning Curve")
plt.legend(); plt.grid(True); plt.show()
```

### 3) Residual analysis

```python
resid = y_test - pred
plt.scatter(pred,resid,alpha=0.7)
plt.axhline(0,color="k",ls="--")
plt.title("Residuals vs Predictions")
plt.xlabel("Predicted"); plt.ylabel("Residual")
plt.grid(True); plt.show()
```

### 4) Feature importance inspection

```python
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", title="Feature Importance")
plt.show()
```

### 5) Sanity check — shuffle labels baseline

```python
from sklearn.utils import shuffle
y_shuff = shuffle(y_train, random_state=0)
rf_shuff = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_train,y_shuff)
pred_shuff = rf_shuff.predict(X_test)
rmse_shuff = mean_squared_error(y_test,pred_shuff,squared=False)
print("Shuffled-label baseline RMSE:", rmse_shuff)
```

---

## Classworks (5) — Skeleton Code

### Classwork 1: Debugging data issues

```python
# ==========================================
# CLASSWORK 1: DATA DEBUGGING
# ==========================================
# Task:
# 1) Load a dataset (or generate synthetic like solar).
# 2) Check for missing values and class/label imbalance.
# 3) Print summary stats.

# df = pd.read_csv("your_dataset.csv")
# print(df.info())
# print(df.isna().sum())
# print(df["target"].value_counts())
```

### Classwork 2: Overfitting diagnosis with learning curves

```python
# ==========================================
# CLASSWORK 2: LEARNING CURVES
# ==========================================
# Task:
# 1) Train a model (RF, Ridge, etc.).
# 2) Compute learning curve.
# 3) Plot train vs validation RMSE; interpret bias vs variance.

# sizes, tr, va = learning_curve(model, X_train, y_train, cv=5,
#                                scoring="neg_root_mean_squared_error",
#                                train_sizes=np.linspace(0.1,1.0,5))
# plt.plot(sizes,-tr.mean(axis=1),label="Train")
# plt.plot(sizes,-va.mean(axis=1),label="CV")
# plt.legend(); plt.grid(True); plt.show()
```

### Classwork 3: Residual analysis

```python
# ==========================================
# CLASSWORK 3: RESIDUALS
# ==========================================
# Task:
# 1) Fit regression model.
# 2) Compute residuals (y_test - y_pred).
# 3) Scatter plot residuals vs predicted.

# model.fit(X_train,y_train)
# pred = model.predict(X_test)
# resid = y_test - pred
# plt.scatter(pred,resid)
# plt.axhline(0,color="k",ls="--")
# plt.grid(True); plt.show()
```

### Classwork 4: Feature importance or SHAP

```python
# ==========================================
# CLASSWORK 4: FEATURE IMPORTANCE
# ==========================================
# Task:
# 1) Train RandomForest or GradientBoosting model.
# 2) Plot feature importances.
# 3) Identify top 2 features driving predictions.

# importances = pd.Series(model.feature_importances_, index=X.columns)
# importances.sort_values().plot(kind="barh")
# plt.show()
```

### Classwork 5: Sanity check with shuffled labels

```python
# ==========================================
# CLASSWORK 5: SHUFFLE BASELINE
# ==========================================
# Task:
# 1) Shuffle target labels.
# 2) Train model on shuffled data.
# 3) Compare RMSE with original model; discuss why high RMSE = sanity check.

# from sklearn.utils import shuffle
# y_shuff = shuffle(y_train, random_state=0)
# model.fit(X_train,y_shuff)
# pred_shuff = model.predict(X_test)
# rmse_shuff = mean_squared_error(y_test,pred_shuff,squared=False)
# print("RMSE (shuffled):", rmse_shuff)
```

---

## Lab: Guided Project Coding

* Pick your **project dataset** (battery, solar, comm signals, IoT, etc.).
* Implement a **baseline model** (Linear, Ridge, RF).
* Debug systematically:

  * Plot learning curves.
  * Residual analysis.
  * Feature importances.
* Write **1-page lab note**: describe bugs/issues discovered (data leaks, overfit, weak features) and fixes attempted.

---

## Project Update (Code Review)

* Submit your code for a **peer/instructor review**:

  * Is preprocessing correct?
  * Are train/test splits proper?
  * Any leakage risks?
  * Is evaluation metric appropriate?
* Outline changes for final version (to be written up in paper draft).

---

### Wrap-up / Homework Challenge

* Extend Classwork 4 with **SHAP values** (`shap` library) for interpretability.
* Write a 5–6 line **debugging diary entry**: “Today my model overfit because… I fixed it by…”

