# Class 8: Model Selection, Hyperparameter Tuning, CV, and Overfitting Control

**Lab:** Tune regression model for **battery life prediction**

### Overview

Today we turn “works on my machine” into “works on *unseen* data.” You’ll set up proper splits, run CV, tune hyperparameters with `GridSearchCV`/`RandomizedSearchCV`, read learning curves, and choose between **bias** (too simple) and **variance** (too wiggly).

### Lecture Notes (pocket edition)

* **Generalization**: performance on new data matters more than train score.
* **Data splits**: **train** (fit), **validation** (tune), **test/holdout** (final check). With CV, validation is simulated by rotating folds.
* **Cross-validation**: K-Fold (regression), **Stratified** for classification. Repeated K-Fold gives stabler estimates.
* **Pipelines**: prevent leakage; scale/transform **inside** CV.
* **Tuning**: Grid (systematic), Randomized (efficient on large spaces).
* **Overfitting control**: regularization (Ridge/Lasso), early stopping (GBMs/NNs), pruning (trees), more data, simpler features.
* **Diagnostics**: learning curves (sample size vs error), validation curves (hyperparameter vs error), residual plots.

---

## Demo Notebook (Colab-style)

### 0) Setup: Synthetic Battery Dataset (self-contained)

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
np.random.seed(8)

# Features that influence remaining battery capacity (Ah) or health (%)
n = 1200
cycles = np.random.randint(50, 1500, size=n)                # cycle count
dod    = np.random.uniform(0.2, 0.9, size=n)                 # depth of discharge
tempC  = np.random.normal(28, 6, size=n).clip(5, 55)         # ambient/storage temp
crate  = np.random.uniform(0.2, 2.0, size=n)                  # charge/discharge C-rate
rimOhm = np.random.normal(45, 10, size=n).clip(15, 120)       # internal resistance (mΩ)

# Nonlinear ground truth with interactions + noise
# capacity_% ~ 100 - a1*cycles - a2*(dod^1.3) - a3*(temp-25)^2 - a4*crate + a5*sqrt(r)
true = (
    100
    - 0.02*cycles
    - 12*(dod**1.3)
    - 0.03*((tempC-25.0)**2)
    - 3.0*crate
    + 0.4*np.sqrt(rimOhm)
)

noise = np.random.normal(0, 2.5, size=n)
capacity_pct = (true + noise).clip(0, 100)

df = pd.DataFrame({
    "cycles": cycles, "dod": dod, "tempC": tempC, "crate": crate, "rimOhm": rimOhm,
    "capacity_pct": capacity_pct
})
df.head()
```

### 1) Train/Holdout split (hold test for the *very* end)

```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=["capacity_pct"])
y = df["capacity_pct"]

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train.shape, X_holdout.shape
```

### 2) Baselines + Cross-validation (Pipelines to avoid leakage)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"  # RMSE (negative, because sklearn maximizes)

pipelines = {
    "Linear": Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]),
    "Ridge":  Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=1.0))]),
    "Lasso":  Pipeline([("sc", StandardScaler()), ("ls", Lasso(alpha=0.01, max_iter=5000))]),
    "RF":     Pipeline([("rf", RandomForestRegressor(n_estimators=300, random_state=42))]),
}

for name, pipe in pipelines.items():
    rmse_scores = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"{name:7s} | CV RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
```

### 3) Grid Search (Ridge/Lasso) + Randomized Search (RandomForest)

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# Ridge grid
ridge_grid = {
    "rg__alpha": np.logspace(-3, 3, 13)
}
ridge_pipe = Pipeline([("sc", StandardScaler()), ("rg", Ridge())])
ridge_cv = GridSearchCV(ridge_pipe, ridge_grid, cv=cv, scoring=scoring, n_jobs=-1)
ridge_cv.fit(X_train, y_train)
print("Ridge best:", ridge_cv.best_params_, "CV RMSE:", -ridge_cv.best_score_)

# Lasso grid
lasso_grid = {
    "ls__alpha": np.logspace(-4, 0, 9)
}
lasso_pipe = Pipeline([("sc", StandardScaler()), ("ls", Lasso(max_iter=10000))])
lasso_cv = GridSearchCV(lasso_pipe, lasso_grid, cv=cv, scoring=scoring, n_jobs=-1)
lasso_cv.fit(X_train, y_train)
print("Lasso best:", lasso_cv.best_params_, "CV RMSE:", -lasso_cv.best_score_)

# RandomForest randomized search
rf_pipe = Pipeline([("rf", RandomForestRegressor(random_state=42))])
rf_dist = {
    "rf__n_estimators": randint(200, 800),
    "rf__max_depth": randint(3, 18),
    "rf__min_samples_split": randint(2, 12),
    "rf__min_samples_leaf": randint(1, 8),
    "rf__max_features": ["auto", "sqrt", 0.5, None],
}
rf_cv = RandomizedSearchCV(rf_pipe, rf_dist, n_iter=30, cv=cv, scoring=scoring, n_jobs=-1, random_state=42)
rf_cv.fit(X_train, y_train)
print("RF best:", rf_cv.best_params_, "CV RMSE:", -rf_cv.best_score_)
```

### 4) Polynomial features + Ridge (bias–variance tradeoff)

```python
poly_pipe = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),   # we'll tune the degree
    ("sc", StandardScaler(with_mean=False)),           # with poly, sparse-ish -> keep with_mean=False
    ("rg", Ridge())
])

poly_grid = {
    "poly__degree": [1, 2, 3],
    "rg__alpha": np.logspace(-3, 2, 10)
}

poly_cv = GridSearchCV(poly_pipe, poly_grid, cv=cv, scoring=scoring, n_jobs=-1)
poly_cv.fit(X_train, y_train)
print("PolyRidge best:", poly_cv.best_params_, "CV RMSE:", -poly_cv.best_score_)
```

### 5) Learning curve (do we need more data or simpler model?)

```python
from sklearn.model_selection import learning_curve
import numpy as np

best_model = rf_cv.best_estimator_  # or poly_cv.best_estimator_
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 6),
    cv=cv, scoring=scoring, n_jobs=-1, shuffle=True, random_state=42
)

train_rmse = -train_scores.mean(axis=1)
val_rmse   = -val_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_rmse, marker="o", label="Train RMSE")
plt.plot(train_sizes, val_rmse, marker="o", label="CV RMSE")
plt.xlabel("Training samples"); plt.ylabel("RMSE")
plt.title("Learning Curve (best model)")
plt.grid(True); plt.legend(); plt.show()
```

### 6) Final evaluation on untouched holdout

```python
final_model = rf_cv.best_estimator_ if -rf_cv.best_score_ <= -poly_cv.best_score_ else poly_cv.best_estimator_
final_model.fit(X_train, y_train)
pred = final_model.predict(X_holdout)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
rmse = mean_squared_error(y_holdout, pred, squared=False)
mae  = mean_absolute_error(y_holdout, pred)

print("HOLDOUT RMSE:", round(rmse, 2), "| MAE:", round(mae, 2))

# Residual plot
resid = y_holdout - pred
plt.figure()
plt.scatter(pred, resid, s=15, alpha=0.7)
plt.axhline(0, color="k", linestyle="--")
plt.title("Residuals vs Predicted (Holdout)")
plt.xlabel("Predicted capacity (%)"); plt.ylabel("Residual")
plt.grid(True); plt.show()
```

### 7) Model interpretation quick-peek (if RF wins)

```python
if "rf" in final_model.named_steps:
    rf = final_model.named_steps["rf"]
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(importances)
    importances.plot(kind="bar", title="RF Feature Importance"); plt.grid(True); plt.show()
```

---

## Classworks (5) — Skeleton Code (students fill the blanks)

### Classwork 1: Compare baselines with proper CV

```python
# ==================================================
# CLASSWORK 1: BASELINES + CV (NO LEAKAGE!)
# ==================================================
# Task:
# 1) Make a Pipeline(StandardScaler, LinearRegression) and Pipeline(StandardScaler, Ridge).
# 2) Evaluate with KFold(n_splits=5, shuffle=True).
# 3) Report mean±std CV RMSE for both.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score

# cv = KFold(n_splits=??, shuffle=True, random_state=0)
# pipe_lr = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
# pipe_rg = Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=1.0))])

# for name, pipe in [("LR", pipe_lr), ("Ridge", pipe_rg)]:
#     rmse = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
#     print(name, "CV RMSE:", rmse.mean(), "±", rmse.std())
```

### Classwork 2: Grid search Ridge/Lasso with param heat

```python
# ==================================================
# CLASSWORK 2: GRIDSEARCH RIDGE/LASSO
# ==================================================
# Task:
# 1) GridSearchCV over alpha (logspace).
# 2) Print best params + CV RMSE.
# 3) Fit on full train and evaluate on holdout.

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ridge_pipe = Pipeline([("sc", StandardScaler()), ("rg", Ridge())])
# ridge_grid = {"rg__alpha": np.logspace(-3, 3, 13)}
# ridge_cv = GridSearchCV(ridge_pipe, ridge_grid, cv=???, scoring="neg_root_mean_squared_error", n_jobs=-1)
# ridge_cv.fit(X_train, y_train)
# print("Ridge best:", ridge_cv.???, "CV RMSE:", -ridge_cv.???)

# best_ridge = ridge_cv.best_estimator_.fit(X_train, y_train)
# hold_rmse = mean_squared_error(y_holdout, best_ridge.predict(X_holdout), squared=False)
# print("Holdout RMSE:", hold_rmse)
```

### Classwork 3: PolynomialFeatures + Ridge validation curve

```python
# ==================================================
# CLASSWORK 3: POLY + RIDGE VALIDATION
# ==================================================
# Task:
# 1) Pipeline(PolynomialFeatures -> StandardScaler -> Ridge).
# 2) Loop degrees in {1,2,3} and alphas in logspace; store CV RMSE in a table.
# 3) Pick the combo minimizing CV RMSE; note overfitting patterns (deg=3 vs deg=1).

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

# degrees = [1,2,3]
# alphas = np.logspace(-3, 2, 10)
# rows = []
# for d in degrees:
#     for a in alphas:
#         pipe = Pipeline([("poly", PolynomialFeatures(degree=d, include_bias=False)),
#                          ("sc", StandardScaler(with_mean=False)),
#                          ("rg", Ridge(alpha=a))])
#         rmse = -cross_val_score(pipe, X_train, y_train, cv=???, scoring="neg_root_mean_squared_error", n_jobs=-1)
#         rows.append({"deg": d, "alpha": a, "cv_rmse": rmse.mean()})
# pd.DataFrame(rows).sort_values("cv_rmse").head()
```

### Classwork 4: RandomizedSearch on RandomForest + feature importance

```python
# ==================================================
# CLASSWORK 4: RANDOMIZED RF + IMPORTANCE
# ==================================================
# Task:
# 1) RandomizedSearchCV over RF hyperparameters (n_estimators, max_depth, min_samples_*).
# 2) Report best params + CV RMSE.
# 3) Fit best on full train, compute holdout RMSE.
# 4) Bar plot feature importances.

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# rf = RandomForestRegressor(random_state=0)
# dist = {
#   "n_estimators": randint(200, 800),
#   "max_depth": randint(3, 18),
#   "min_samples_split": randint(2, 12),
#   "min_samples_leaf": randint(1, 8),
# }
# rf_cv = RandomizedSearchCV(rf, dist, n_iter=30, cv=???, scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=0)
# rf_cv.fit(X_train, y_train)
# print("Best:", rf_cv.best_params_, "CV RMSE:", -rf_cv.best_score_)
# best_rf = rf_cv.best_estimator_.fit(X_train, y_train)
# print("Holdout RMSE:", mean_squared_error(y_holdout, best_rf.predict(X_holdout), squared=False))

# pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).plot(kind="bar")
# plt.title("RF Feature Importance"); plt.grid(True); plt.show()
```

### Classwork 5: Learning curve & overfitting diagnosis

```python
# ==================================================
# CLASSWORK 5: LEARNING CURVE DIAGNOSIS
# ==================================================
# Task:
# 1) Pick your best model (from CW2–CW4).
# 2) Plot learning curve (train_sizes 10%→100%, KFold CV).
# 3) Interpret: high bias vs high variance? Suggest a fix.

from sklearn.model_selection import learning_curve

# best_model = ???  # your best estimator
# sizes, tr, va = learning_curve(best_model, X_train, y_train, train_sizes=np.linspace(0.1,1.0,6),
#                                cv=???, scoring="neg_root_mean_squared_error", n_jobs=-1, shuffle=True, random_state=0)
# plt.figure()
# plt.plot(sizes, -tr.mean(axis=1), marker="o", label="Train RMSE")
# plt.plot(sizes, -va.mean(axis=1), marker="o", label="CV RMSE")
# plt.xlabel("Training samples"); plt.ylabel("RMSE")
# plt.title("Learning Curve (Your Best Model)")
# plt.grid(True); plt.legend(); plt.show()

# # Write 2–3 lines: are curves far apart (variance)? both high (bias)?
```

---

## Lab: Tune a Regression Model for **Battery Life Prediction**

**Goal:** Predict `capacity_pct` from `cycles, dod, tempC, crate, rimOhm` with the **lowest holdout RMSE** while keeping the model explainable.

**Suggested path (already scaffolded in the demo):**

1. Start with **Ridge** grid; record best alpha + CV RMSE.
2. Try **PolynomialFeatures + Ridge**; watch for overfitting (degree ↑).
3. Try **RandomForest** randomized search; compare holdout RMSE vs PolyRidge.
4. Plot **residuals**; check patterns vs `cycles` and `tempC`.
5. Create a short **model card** (Markdown cell): data, features, best hyperparameters, CV RMSE, holdout RMSE, limitations, next steps (e.g., add storage time, humidity, calendar aging features).

---

## Project Topic Prompts (pick one and scope it)

* **Battery health monitoring:** Use real logs (cycles, DoD, temp profiles) to forecast **remaining useful life (RUL)**; compare Ridge vs RF vs Gradient Boosting.
* **Solar-battery microgrid:** Predict **daily discharge depth** from weather + load; evaluate regularization impact.
* **Sensor drift compensation:** Calibrate raw sensor output against reference; compare Polynomial Ridge vs Tree-based models.
* **Charging optimization:** Model how **C-rate** and **thermal conditions** impact capacity fade; do controlled what-ifs.

---

### Wrap-up / Homework Challenge

* **Nested CV** sanity check: outer 5-fold for generalization estimate, inner CV for tuning. Report outer-fold RMSE mean±std.
* **Regularization report:** for Ridge, plot CV RMSE vs `log10(alpha)`; annotate the “sweet spot” and explain *why* it beats both under- and over-regularization.

