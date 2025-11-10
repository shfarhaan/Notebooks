# Class 6: Classification — Logistic Regression & Decision Trees (Fault Detection)

### Overview

Regression estimates a value; **classification** decides a label. Today’s mission: given basic features from a signal (mean, std, gradients, threshold flags), classify whether a segment is **Normal** or **Fault** (and optionally which fault).

### Lecture Notes (pocket edition)

* **Binary classification**: labels {0,1}.
* **Logistic Regression**: linear decision boundary; outputs probability via sigmoid.
* **Decision Tree**: axis-aligned splits; handles nonlinearity; watch for overfitting → limit depth.
* **Metrics**: accuracy (overall), precision (purity of predicted positives), recall (how many positives caught), F1 (balance), ROC-AUC (ranking quality).
* **Confusion matrix**: TP/FP/FN/TN — vital when false alarms vs misses have different costs.

---

## Demo Notebook (Colab-style)

### 0) Setup: create a toy “fault vs normal” dataset (self-contained)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, RocCurveDisplay
)

np.random.seed(123)

# Simulate 2 classes using simple stats features from windows
n = 600
# Features: mean voltage, std voltage, slope proxy, threshold flag
mean_normal  = np.random.normal(3.20, 0.02, n//2)
std_normal   = np.random.normal(0.03, 0.01, n//2).clip(0.005, None)
slope_normal = np.random.normal(0.00, 0.01, n//2)
flag_normal  = (mean_normal > 3.6).astype(int)  # mostly 0

mean_fault   = np.random.normal(3.35, 0.06, n//2)
std_fault    = np.random.normal(0.10, 0.03, n//2).clip(0.01, None)
slope_fault  = np.random.normal(0.03, 0.02, n//2)
flag_fault   = (mean_fault > 3.6).astype(int)  # occasional 1s

X = np.column_stack([
    np.concatenate([mean_normal, mean_fault]),
    np.concatenate([std_normal,  std_fault]),
    np.concatenate([slope_normal, slope_fault]),
    np.concatenate([flag_normal,  flag_fault])
]).astype(float)

y = np.array([0]*(n//2) + [1]*(n//2))  # 0=Normal, 1=Fault

df = pd.DataFrame(X, columns=["mean_v","std_v","slope","flag_high"])
df["label"] = y
df.head()
```

### 1) Split, scale (for LR), train Logistic Regression

```python
X = df[["mean_v","std_v","slope","flag_high"]].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=200, class_weight=None)  # try class_weight="balanced" if imbalanced
logreg.fit(X_train_s, y_train)

y_prob_lr = logreg.predict_proba(X_test_s)[:,1]
y_pred_lr = (y_prob_lr >= 0.5).astype(int)

print("LR accuracy:", accuracy_score(y_test, y_pred_lr))
print("LR precision:", precision_score(y_test, y_pred_lr))
print("LR recall:", recall_score(y_test, y_pred_lr))
print("LR F1:", f1_score(y_test, y_pred_lr))
print("LR ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
```

### 2) Train a Decision Tree (nonlinear, interpretable)

```python
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

y_prob_dt = tree.predict_proba(X_test)[:,1]
y_pred_dt = tree.predict(X_test)

print("DT accuracy:", accuracy_score(y_test, y_pred_dt))
print("DT precision:", precision_score(y_test, y_pred_dt))
print("DT recall:", recall_score(y_test, y_pred_dt))
print("DT F1:", f1_score(y_test, y_pred_dt))
print("DT ROC-AUC:", roc_auc_score(y_test, y_prob_dt))
```

### 3) Confusion matrices + ROC curves

```python
fig, ax = plt.subplots(1,2, figsize=(10,4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr), display_labels=["Normal","Fault"]).plot(ax=ax[0], colorbar=False)
ax[0].set_title("Logistic Regression")

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt), display_labels=["Normal","Fault"]).plot(ax=ax[1], colorbar=False)
ax[1].set_title("Decision Tree")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_predictions(y_test, y_prob_lr)
plt.title("ROC — Logistic Regression")
plt.grid(True); plt.show()

RocCurveDisplay.from_predictions(y_test, y_prob_dt)
plt.title("ROC — Decision Tree")
plt.grid(True); plt.show()
```

### 4) Threshold tuning (precision/recall trade-off)

```python
def evaluate_threshold(y_true, y_scores, thresh):
    y_hat = (y_scores >= thresh).astype(int)
    return {
        "thr": thresh,
        "acc": accuracy_score(y_true, y_hat),
        "prec": precision_score(y_true, y_hat, zero_division=0),
        "rec": recall_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat)
    }

thresholds = np.linspace(0.1, 0.9, 9)
rows = [evaluate_threshold(y_test, y_prob_lr, t) for t in thresholds]
pd.DataFrame(rows)
```

### 5) Peek inside the tree (feature splits)

```python
plt.figure(figsize=(10,6))
plot_tree(tree, feature_names=["mean_v","std_v","slope","flag_high"], class_names=["Normal","Fault"], filled=True, rounded=True)
plt.show()

print("Tree feature importances:", dict(zip(["mean_v","std_v","slope","flag_high"], tree.feature_importances_)))
print("LR coefficients:", dict(zip(["mean_v","std_v","slope","flag_high"], logreg.coef_[0])))
```

---

## Classworks (5) — Skeleton Code (students fill in)

### Classwork 1: Build & split your own dataset

```python
# ==========================================
# CLASSWORK 1: CUSTOM DATASET + SPLIT
# ==========================================
# Task:
# 1) Create a small DataFrame with columns: mean_v, std_v, slope, flag_high, label.
# 2) Do a stratified train/test split (test_size=0.3).
# 3) Print class counts in train and test.

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

# df = pd.DataFrame({
#     "mean_v":  ???,
#     "std_v":   ???,
#     "slope":   ???,
#     "flag_high": ???,
#     "label":   ???  # 0/1
# })
# X = df[["mean_v","std_v","slope","flag_high"]].values
# y = df["label"].values
# X_train, X_test, y_train, y_test = train_test_split(???, ???, test_size=0.3, stratify=???, random_state=0)
# print("Train counts:", np.bincount(y_train))
# print("Test counts :", np.bincount(y_test))
```

### Classwork 2: Logistic Regression with scaling + metrics

```python
# ==========================================
# CLASSWORK 2: LOGISTIC REGRESSION + METRICS
# ==========================================
# Task:
# 1) Scale features with StandardScaler.
# 2) Fit LogisticRegression (max_iter=200).
# 3) Compute accuracy, precision, recall, F1, ROC-AUC on TEST.
# 4) Print classification_report.

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# scaler = StandardScaler()
# X_train_s = scaler.???.???(X_train)
# X_test_s  = scaler.???(X_test)

# clf = LogisticRegression(max_iter=200)
# clf.???(X_train_s, y_train)

# y_prob = clf.???(X_test_s)[:,1]
# y_pred = (y_prob >= 0.5).astype(int)

# print("ACC:", ???(y_test, y_pred))
# print("PREC:", ???(y_test, y_pred))
# print("REC:",  ???(y_test, y_pred))
# print("F1:",   ???(y_test, y_pred))
# print("AUC:",  ???(y_test, y_prob))
# print(classification_report(y_test, y_pred))
```

### Classwork 3: Decision Tree depth sweep

```python
# ==========================================
# CLASSWORK 3: TREE DEPTH SWEEP
# ==========================================
# Task:
# 1) Train DecisionTreeClassifier with max_depth in {2,3,4,5,6}.
# 2) For each depth, compute test accuracy and F1; store in a table.
# 3) Pick the "best" depth (justify via F1).

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# results = []
# for d in [2,3,4,5,6]:
#     dt = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_train, y_train)
#     y_hat = dt.predict(X_test)
#     results.append({"depth": d, "acc": ???(y_test, y_hat), "f1": ???(y_test, y_hat)})
# import pandas as pd
# pd.DataFrame(results)
```

### Classwork 4: Confusion matrix + ROC for your chosen model

```python
# ==========================================
# CLASSWORK 4: CM + ROC
# ==========================================
# Task:
# 1) Take your best model from CW2 or CW3.
# 2) Plot confusion matrix and ROC curve.
# 3) Briefly comment: is the model conservative or aggressive on positives?

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# y_prob_best = ???  # predicted probabilities for class 1
# y_pred_best = (y_prob_best >= 0.5).astype(int)

# ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_best), display_labels=["Normal","Fault"]).plot(colorbar=False)
# plt.title("Confusion Matrix"); plt.grid(False); plt.show()

# RocCurveDisplay.from_predictions(y_test, y_prob_best)
# plt.title("ROC Curve"); plt.grid(True); plt.show()
```

### Classwork 5: Threshold tuning table

```python
# ==========================================
# CLASSWORK 5: THRESHOLD TUNING
# ==========================================
# Task:
# 1) For thresholds in np.arange(0.2, 0.9, 0.1), compute precision, recall, F1.
# 2) Make a small DataFrame "tuning" with columns thr, prec, rec, f1.
# 3) Choose a threshold that prioritizes recall (explain in one sentence).

import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# thresholds = np.arange(0.2, 0.9, 0.1)
# rows = []
# for thr in thresholds:
#     y_hat = (y_prob_best >= thr).astype(int)
#     rows.append({
#         "thr": thr,
#         "prec": ???(y_test, y_hat, zero_division=0),
#         "rec":  ???(y_test, y_hat),
#         "f1":   ???(y_test, y_hat),
#     })
# tuning = pd.DataFrame(rows)
# tuning
```

---

### Wrap-up / Homework Challenge

* Re-simulate with **class imbalance** (e.g., 80% Normal, 20% Fault). Compare LR with and without `class_weight="balanced"`; compare tree depths. Discuss which metric you’d report to plant ops if **missing a fault** is very costly (hint: recall / FN).
* Bonus: add a **third class** (e.g., “Transient”) and try a **multiclass** Decision Tree; report macro-F1.

