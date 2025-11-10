# Class 5: ML Basics — Linear Regression (Calibration & Curve Fitting)

### Overview

Linear regression is the lab scale of ML: you put in an input, it gives a prediction with a straight-line relationship. In EE/IoT land, you’ll use it to calibrate sensors, correct drift, or estimate quantities that are hard to measure directly.

---

### Lecture Notes (short, teachable)

* **Goal**: learn a mapping ( y \approx \beta_0 + \beta_1 x ) (or multi-feature ( \mathbf{y} \approx \mathbf{X}\boldsymbol{\beta} )).
* **Train / test split**: measure generalization honestly.
* **Metrics**: MAE (average absolute error), MSE (squared), (R^2) (explained variance).
* **Residuals**: ( r_i = y_i - \hat{y}_i ); plot them — they confess model sins (nonlinearity, heteroscedasticity, outliers).
* **Common mistakes**: fitting on test data, leaking future info, ignoring units/scales/outliers.

EE hooks:

* Calibrate thermistor: temperature → resistance (near-linear in a small range).
* Battery health: cycle count/features → capacity drop (piecewise/approx linear).
* Sensor cross-calibration: raw reading → true reference value.

---

## Demo Notebook (Colab-style cells)

### 0) Imports + synthetic calibration dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

# Synthetic "sensor calibration": true_y = 2.5 + 0.8*x + noise
n = 120
x = np.linspace(0, 50, n)                              # e.g., temperature in °C
noise = np.random.normal(0, 1.2, size=n)               # measurement noise
y_true = 2.5 + 0.8 * x                                 # ideal mapping
y = y_true + noise                                     # observed reading to be predicted

df = pd.DataFrame({"temp_C": x, "reading": y, "ideal": y_true})
df.head()
```

### 1) Train/test split and model fit

```python
X = df[["temp_C"]].values
y = df["reading"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept (β0):", model.intercept_)
print("Slope (β1):", model.coef_[0])
```

### 2) Evaluate — metrics + predicted line

```python
y_pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
r2  = r2_score(y_test, y_pred_test)

print(f"MAE={mae:.3f}, MSE={mse:.3f}, R^2={r2:.3f}")

# Visualize fit against all data
x_grid = np.linspace(df["temp_C"].min(), df["temp_C"].max(), 200).reshape(-1, 1)
y_line = model.predict(x_grid)

plt.figure(figsize=(6,4))
plt.scatter(df["temp_C"], df["reading"], s=20, label="data", alpha=0.7)
plt.plot(x_grid, y_line, label="fitted line", linewidth=2)
plt.title("Linear Regression Fit (Sensor Calibration)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Reading (units)")
plt.grid(True)
plt.legend()
plt.show()
```

### 3) Residual analysis (plot + quick checks)

```python
# Residuals on the test set
resid = y_test - y_pred_test

plt.figure(figsize=(6,4))
plt.scatter(y_pred_test, resid, s=25)
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.title("Residuals vs Predicted (Test)")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.grid(True)
plt.show()

print("Residual mean (should be ~0):", np.mean(resid))
print("Residual std:", np.std(resid))
```

### 4) Multi-feature quick demo (add a small linear drift feature)

```python
# Add a second feature that could represent another sensor or derived feature
df["drift"] = 0.02 * df["temp_C"]  # tiny drift correlated with temp

X2 = df[["temp_C", "drift"]].values
y2 = df["reading"].values

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=0)

model2 = LinearRegression().fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

print("Coefs (β):", model2.coef_, "| Intercept:", model2.intercept_)
print("R^2 (2 features):", r2_score(y2_test, y2_pred))
```

---

## Classworks (5) — Skeleton Code (students fill the blanks)

### Classwork 1: Fit a line and report metrics

```python
# ==========================================
# CLASSWORK 1: SIMPLE LINEAR REGRESSION
# ==========================================
# Task:
# 1) Build a synthetic dataset: y = a + b*x + noise (choose a, b).
# 2) Split train/test.
# 3) Fit LinearRegression, compute MAE, MSE, R^2 on test.
# 4) Plot data + fitted line.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(0)

# x = np.linspace(???, ???, ???).reshape(-1, 1)
# noise = np.random.normal(0, ???, size=x.shape[0])
# y = ??? + ??? * x.flatten() + noise

# X_train, X_test, y_train, y_test = train_test_split(???, ???, test_size=0.2, random_state=0)

# model = LinearRegression()
# model.fit(???, ???)

# y_pred = model.predict(???)
# mae = mean_absolute_error(???, ???)
# mse = mean_squared_error(???, ???)
# r2  = r2_score(???, ???)
# print(f"MAE={mae:.3f}, MSE={mse:.3f}, R^2={r2:.3f}")

# # plot
# x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
# y_line = model.predict(x_grid)
# plt.figure()
# plt.scatter(x, y, s=20, label="data", alpha=0.7)
# plt.plot(x_grid, y_line, label="fit", linewidth=2)
# plt.title("Simple Linear Regression")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(True); plt.legend(); plt.show()
```

### Classwork 2: Residuals check + assumptions

```python
# ==========================================
# CLASSWORK 2: RESIDUAL DIAGNOSTICS
# ==========================================
# Task:
# 1) Reuse your model from CW1.
# 2) Compute residuals on TEST set: r = y_test - y_pred.
# 3) Plot residuals vs predicted.
# 4) Print mean and std of residuals.

# resid = ??? - ???
# import numpy as np, matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(???, resid, s=25)
# plt.axhline(0, color="k", linestyle="--")
# plt.title("Residuals vs Predicted")
# plt.xlabel("Predicted"); plt.ylabel("Residual")
# plt.grid(True); plt.show()

# print("Residual mean:", np.mean(???))
# print("Residual std:", np.std(???))
```

### Classwork 3: Add an irrelevant feature (sanity test)

```python
# =======================================================
# CLASSWORK 3: ADD AN IRRELEVANT FEATURE (NOISE FEATURE)
# =======================================================
# Task:
# 1) Generate a random feature z (same length as x), uncorrelated with y.
# 2) Fit LinearRegression on [x, z].
# 3) Compare R^2 with 1-feature model. Does it improve meaningfully?

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# z = np.random.randn(x.shape[0]).reshape(-1, 1)
# X2 = np.hstack([x, z])
# X2_train, X2_test, y_train, y_test = train_test_split(???, ???, test_size=0.2, random_state=0)

# model1 = LinearRegression().fit(X_train, y_train)   # from CW1
# model2 = LinearRegression().fit(X2_train, y_train)

# r2_1 = r2_score(y_test, model1.predict(X_test))
# r2_2 = r2_score(y_test, model2.predict(X2_test))
# print("R^2 (1 feature):", r2_1, " | R^2 (x + noise z):", r2_2)
```

### Classwork 4: Mild nonlinearity — where linear fails

```python
# =======================================================
# CLASSWORK 4: MILD NONLINEARITY CHECK
# =======================================================
# Task:
# 1) Create data: y = 1.0 + 0.2*x + 0.02*x^2 + noise (slight curve).
# 2) Fit linear model on x only.
# 3) Plot residuals vs x: do you see structure (curve)?
# 4) Optional: add x^2 as a feature and show improvement in R^2.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(1)
# x2 = np.linspace(???, ???, ???).reshape(-1, 1)
# noise = np.random.normal(0, 0.5, size=x2.shape[0])
# y2 = 1.0 + 0.2*x2.flatten() + 0.02*(x2.flatten()**2) + noise

# # Linear (wrong) model
# X_lin = x2
# model_lin = LinearRegression().fit(X_lin, y2)
# y2_pred_lin = model_lin.predict(X_lin)

# plt.figure()
# plt.scatter(x2, y2 - y2_pred_lin, s=20)
# plt.axhline(0, color="k", linestyle="--")
# plt.title("Residuals vs x (nonlinearity visible?)")
# plt.xlabel("x")
# plt.ylabel("residual")
# plt.grid(True); plt.show()

# # Optional polynomial fix (manual feature)
# X_poly = np.hstack([x2, x2**2])
# model_poly = LinearRegression().fit(X_poly, y2)
# print("R^2 linear:", r2_score(y2, y2_pred_lin))
# print("R^2 poly  :", r2_score(y2, model_poly.predict(X_poly)))
```

### Classwork 5: Mini calibration case (CSV)

```python
# =======================================================
# CLASSWORK 5: MINI CALIBRATION FROM CSV
# =======================================================
# Task:
# 1) Create or load 'calib.csv' with columns raw_reading, true_value (at least 50 rows).
# 2) Fit LinearRegression to map raw_reading -> true_value.
# 3) Report MAE, MSE, R^2 on test split.
# 4) Plot: scatter(raw_reading, true_value) + fitted line.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv("calib.csv")  # or generate synthetic then save
# X = df[["raw_reading"]].values
# y = df["true_value"].values

# X_train, X_test, y_train, y_test = train_test_split(???, ???, test_size=0.25, random_state=0)
# model = LinearRegression().fit(???, ???)
# y_pred = model.predict(???)

# print("MAE:", ???(???, ???))
# print("MSE:", ???(???, ???))
# print("R^2:", ???(???, ???))

# # Plot
# x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
# y_line = model.predict(x_grid)
# plt.figure()
# plt.scatter(X, y, s=20, alpha=0.7, label="data")
# plt.plot(x_grid, y_line, "r", label="fit")
# plt.xlabel("raw_reading")
# plt.ylabel("true_value")
# plt.title("Calibration Fit")
# plt.grid(True); plt.legend(); plt.show()
```

---

### Wrap-up / Homework Challenge

* **Drift-aware calibration**: Add a second feature (e.g., ambient temperature) to your CSV and compare 1-feature vs 2-feature models via (R^2) and MAE. Write 3 lines interpreting whether the extra feature is genuinely useful or just noise.
* Bonus: save a **model card** (Markdown cell or text file) summarizing data source, splits, metrics, known limitations, and next steps (e.g., try polynomial features or regularization later).

