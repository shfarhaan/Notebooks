# Class 13: Anomaly Detection + Forecasting (ARIMA & Isolation Forest)

**Lab:** Detect anomalies in power-system data; **Project update:** model selection

---

### Overview

Two complementary lenses on time series:

* **Forecast-then-flag (ARIMA):** predict the next value and mark points where the **residual** (actual − forecast) is unusually large. Great for *temporal* structure.
* **Feature-then-flag (Isolation Forest):** compute rolling features (mean, std, slope, etc.) and flag outliers in that multivariate space. Great for *shape* anomalies across windows.

Together, they’re peanut butter + jelly for power signals.

---

### Lecture Notes (pocket guide)

* **Stationarity:** ARIMA expects stable mean/variance; use **differencing (I)**, **ACF/PACF** to choose **p (AR)** and **q (MA)**.
* **ARIMA(p,d,q):** AR (auto-regressive), I (differencing), MA (moving average). **SARIMA** adds seasonality (P,D,Q,s).
* **Forecast-based anomalies:** large residuals or high prediction intervals breach → anomaly candidates.
* **Isolation Forest (IF):** isolates points by random splits; fewer splits → more anomalous. Use rolling windows to create features.
* **Evaluation:** if you know some injected anomalies, compute precision/recall; otherwise, inspect with plots and domain constraints.

---

## Demo Notebook (Colab-style cells)

### 0) Make a synthetic power load series with anomalies (daily pattern + spikes/drops)

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
np.random.seed(13)

# 60 days of hourly load (1440 samples for minute-level; we'll use hourly for clarity)
hours = 24 * 120  # 120 days
t = np.arange(hours)
# Base: daily sinusoidal demand + slow trend + noise
daily = 50 + 10*np.sin(2*np.pi*t/24)
trend = 0.02*(t/24)  # gentle growth per day
noise = np.random.normal(0, 1.5, size=hours)
y = daily + trend + noise

# Inject anomalies: spikes and sudden drops (10 total)
anom_idx = np.random.choice(np.arange(48, hours-48), size=10, replace=False)
y_anom = y.copy()
y_anom[anom_idx[:5]] += np.random.uniform(15, 25, size=5)   # spikes
y_anom[anom_idx[5:]] -= np.random.uniform(15, 25, size=5)   # dips

ts = pd.Series(y_anom, index=pd.date_range("2024-01-01", periods=hours, freq="H"), name="load_MW")
labels = pd.Series(0, index=ts.index)
labels.iloc[anom_idx] = 1  # ground truth labels (1 = anomaly)

plt.figure(figsize=(10,3))
ts.plot(lw=1)
plt.title("Synthetic Power Load with Injected Anomalies")
plt.ylabel("Load (MW)"); plt.grid(True); plt.show()
```

### 1) Forecast-then-flag with (S)ARIMA (seasonal daily cycle)

```python
# We'll use SARIMA with daily seasonality s=24
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Split: first 100 days train, last 20 days test
split = ts.index[int(len(ts)* (100/120))]
train, test = ts[:split], ts[split:]

# Simple SARIMA order guess (p,d,q) x (P,D,Q,24)
order = (2,1,2)
seasonal_order = (1,1,1,24)

model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)

# Forecast the test horizon
fc = res.get_forecast(steps=len(test))
pred = fc.predicted_mean
ci = fc.conf_int(alpha=0.05)  # 95% PI

ax = test.plot(label="actual", figsize=(10,3))
pred.plot(ax=ax, label="forecast")
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color="gray", alpha=0.2, label="95% PI")
plt.title("SARIMA Forecast on Holdout")
plt.ylabel("Load (MW)"); plt.legend(); plt.grid(True); plt.show()

# Residuals on the whole series using one-step ahead (dynamic) forecasting
full = pd.concat([train, test])
fit_full = SARIMAX(full, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fitted = fit_full.fittedvalues  # in-sample one-step predictions
resid = full - fitted
resid.name = "residual"

plt.figure(figsize=(10,3))
resid.plot(lw=0.8)
plt.axhline(0, color="k", ls="--", lw=1)
plt.title("Residuals (One-step Ahead)")
plt.grid(True); plt.show()
```

### 2) Residual-based anomaly score (z-score or PI breach)

```python
# Z-score of residuals using rolling std for local adaptivity
roll_std = resid.rolling(24*7, min_periods=24).std()  # 1-week window
z = (resid / (roll_std + 1e-6)).abs()

# Flag anomalies: z > 3 or actual outside 95% PI (on test range)
z_thresh = 3.0
flag_z = (z > z_thresh).astype(int)

# Precision/Recall vs ground truth (on full series since we labeled all)
from sklearn.metrics import precision_score, recall_score, f1_score
print("ARIMA residual Z>3 | Precision:", precision_score(labels, flag_z, zero_division=0),
      "Recall:", recall_score(labels, flag_z),
      "F1:", f1_score(labels, flag_z))

# Visualize signals and anomaly markers
ax = ts.plot(figsize=(10,3), lw=1, label="load")
(ts[flag_z.astype(bool)]).plot(ax=ax, marker="o", ls="none", ms=5, label="ARIMA anomalies")
plt.title("Anomalies by Forecast Residuals")
plt.legend(); plt.grid(True); plt.show()
```

### 3) Feature-then-flag with Isolation Forest (windowed features)

```python
from sklearn.ensemble import IsolationForest

# Build rolling-window features (e.g., 24-hour window)
w = 24
df_feat = pd.DataFrame({
    "mean": ts.rolling(w).mean(),
    "std": ts.rolling(w).std(),
    "p2p": ts.rolling(w).max() - ts.rolling(w).min(),
    "slope": ts.diff().rolling(w).mean(),   # average first difference
})
df_feat = df_feat.dropna()

# Fit Isolation Forest on features (unsupervised)
if_model = IsolationForest(n_estimators=300, contamination=0.01, random_state=0)
if_scores = if_model.fit_predict(df_feat)  # -1 = anomaly, 1 = normal
flag_if = pd.Series((if_scores == -1).astype(int), index=df_feat.index)

# Align with labels (intersecting index)
y_true = labels.reindex(df_feat.index).fillna(0).astype(int)
print("IsolationForest | Precision:", precision_score(y_true, flag_if, zero_division=0),
      "Recall:", recall_score(y_true, flag_if),
      "F1:", f1_score(y_true, flag_if))

ax = ts.plot(figsize=(10,3), lw=1, label="load")
(ts[flag_if.astype(bool)]).plot(ax=ax, marker="x", ls="none", ms=5, label="IF anomalies")
plt.title("Anomalies by Isolation Forest")
plt.legend(); plt.grid(True); plt.show()
```

### 4) Compare methods and combine (logical OR to be conservative)

```python
# Align ARIMA flags to IF index
flag_z_aligned = flag_z.reindex(df_feat.index).fillna(0).astype(int)
flag_union = ((flag_if == 1) | (flag_z_aligned == 1)).astype(int)
print("Union (IF ∪ ARIMA) | Precision:", precision_score(y_true, flag_union, zero_division=0),
      "Recall:", recall_score(y_true, flag_union),
      "F1:", f1_score(y_true, flag_union))

ax = ts.plot(figsize=(10,3), lw=1, label="load")
(ts[flag_union.astype(bool)]).plot(ax=ax, marker="o", ls="none", ms=5, label="Union anomalies")
plt.title("Combined Anomaly Flags")
plt.legend(); plt.grid(True); plt.show()
```

---

## Classworks (5) — Skeleton Code (students fill in)

### Classwork 1: Visual sanity & labels

```python
# ==========================================
# CLASSWORK 1: CREATE/LOAD SERIES + LABELS
# ==========================================
# Task:
# 1) Create a time series 'ts' (hourly, >= 60 days) with daily pattern + noise.
# 2) Inject at least 8 anomalies (spikes/drops) and store indices in 'anom_idx'.
# 3) Make a 'labels' Series aligned to ts (1=anomaly else 0).
# 4) Plot ts + mark anomalies.

# ts = pd.Series(..., index=pd.date_range(..., freq="H"))
# anom_idx = [...]
# labels = pd.Series(0, index=ts.index); labels.iloc[anom_idx] = 1
# ax = ts.plot(figsize=(10,3), lw=1)
# ts.iloc[anom_idx].plot(ax=ax, ls="none", marker="o", ms=5)
# plt.grid(True); plt.show()
```

### Classwork 2: Fit SARIMA and compute residual z-scores

```python
# ==========================================
# CLASSWORK 2: (S)ARIMA RESIDUALS
# ==========================================
# Task:
# 1) Choose SARIMA orders (p,d,q)x(P,D,Q,24) and fit on train split.
# 2) Compute one-step-ahead fitted values on full series.
# 3) Residuals = actual - fitted. Compute rolling std and z-scores.
# 4) Flag z > 3 as anomaly.

# from statsmodels.tsa.statespace.sarimax import SARIMAX
# order = (??, ??, ??); seasonal_order = (??, ??, ??, 24)
# res = SARIMAX(ts[:split], order=order, seasonal_order=seasonal_order).fit(disp=False)
# fit_full = SARIMAX(ts, order=order, seasonal_order=seasonal_order).fit(disp=False)
# fitted = fit_full.fittedvalues
# resid = ts - fitted
# z = (resid / (resid.rolling(24*7, min_periods=24).std() + 1e-6)).abs()
# flag_arima = (z > 3).astype(int)
```

### Classwork 3: Isolation Forest on rolling features

```python
# ==========================================
# CLASSWORK 3: ISOLATION FOREST
# ==========================================
# Task:
# 1) Build rolling features (mean, std, p2p, slope) with window=24.
# 2) Fit IsolationForest(contamination=0.02) on features.
# 3) Get anomaly flags (-1→1).
# 4) Compute precision/recall vs labels (align indexes).

# w = 24
# df_feat = pd.DataFrame({
#   "mean": ts.rolling(w).mean(),
#   "std": ts.rolling(w).std(),
#   "p2p": ts.rolling(w).max() - ts.rolling(w).min(),
#   "slope": ts.diff().rolling(w).mean(),
# }).dropna()
# from sklearn.ensemble import IsolationForest
# if_model = IsolationForest(n_estimators=300, contamination=0.02, random_state=0)
# flags_if = pd.Series((if_model.fit_predict(df_feat)==-1).astype(int), index=df_feat.index)
# y_true = labels.reindex(df_feat.index).fillna(0).astype(int)
# # print precision/recall/F1
```

### Classwork 4: Combine detectors & tune thresholds

```python
# ==========================================
# CLASSWORK 4: COMBINE & TUNE
# ==========================================
# Task:
# 1) Make union and intersection of ARIMA and IF flags.
# 2) Sweep z-threshold from {2.0, 2.5, 3.0, 3.5}.
# 3) For each threshold, compute Precision/Recall/F1 of union.
# 4) Choose a setting prioritizing Recall (justify).

# z_values = [2.0, 2.5, 3.0, 3.5]
# rows = []
# for zt in z_values:
#     flag_arima_zt = ((z > zt).astype(int)).reindex(df_feat.index).fillna(0).astype(int)
#     union = ((flag_arima_zt==1) | (flags_if==1)).astype(int)
#     # compute P/R/F1 vs y_true
#     rows.append({"z": zt, "prec": ..., "rec": ..., "f1": ...})
# pd.DataFrame(rows)
```

### Classwork 5: Optional SARIMA hyperparam sweep (quick search)

```python
# ==========================================
# CLASSWORK 5: SARIMA QUICK SWEEP
# ==========================================
# Task:
# 1) Try small grids for p,d,q in {0,1,2}, P,D,Q in {0,1} with s=24 (keep combinations reasonable).
# 2) Fit on train, compute forecast RMSE on validation slice.
# 3) Pick best order and refit on full series; report anomaly F1 with z=3.

# from itertools import product
# pdq = [(p,d,q) for p,d,q in product([0,1,2],[0,1],[0,1,2])]
# PDQ = [(P,D,Q,24) for P,D,Q in product([0,1],[0,1],[0,1])]
# # loop (keep to small subset to finish fast)
# # track validation RMSE; store best (order, seasonal_order)
```

---

## Lab: Detect Anomalies in Power-System Data

**Goal:** Use **both** approaches on your series:

1. **SARIMA residual z-score** (threshold tune): produce an anomaly timeline and precision/recall vs labels.
2. **Isolation Forest on rolling features**: same metrics and a plot.
3. **Compare & combine**: union vs intersection; pick a policy for plant ops (favor catching all true anomalies → higher recall, or minimizing false alarms → higher precision).
4. Save a one-page lab note: dataset, chosen params, plots, metrics, and an operational recommendation.

---

## Project Update (Model Selection)

* Write 5–8 lines addressing:

  * Which detector aligns with your project’s tolerance (misses vs false alarms)?
  * If your data has strong seasonality, lean **SARIMA/Prophet** for explainability; if multivariate shape matters, lean **IF / One-Class SVM / Autoencoder**.
  * What features or external regressors (weather, calendar) would boost reliability?
  * Next step: small **pilot threshold** and on-call workflow (who investigates a flag, within how long).

---

### Wrap-up / Homework Challenge

* Add a **prediction-interval breach** rule: flag anomalies when actual is **outside 99% PI** from SARIMA; compare to z-score.
* Try **contamination tuning** for IF in {0.005, 0.01, 0.02}; plot precision-recall trade-off and choose a setting for a high-risk substation.
