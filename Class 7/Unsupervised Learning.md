# Class 7: Unsupervised Learning — K-means & PCA (Sensor Pattern Discovery)

### Overview

When labels are unknown (which is most of real life), clustering and dimensionality reduction help you “see the shape” of data. We’ll standardize features, try different values of **k**, score clusters, and use **PCA** to project high-dimensional sensor features to 2D for intuition and plotting.

### Lecture Notes (short + teachable)

* **Unsupervised learning**: find structure without labels.
* **K-means**: partitions points into k clusters by minimizing within-cluster variance. Sensitive to scale → **always standardize** features. Use **elbow** (inertia) & **silhouette** to choose k.
* **PCA**: rotates to orthogonal axes capturing maximum variance; good for visualization and noise reduction. Watch explained variance ratio.
* **Workflow**:

  1. Build features (means, stds, slopes, frequency energy, etc.)
  2. **Scale** → **PCA** (optional for viz) → **K-means**
  3. Inspect clusters; interpret with domain sense (EE: operating states, fault regimes, sensor placement effects)

---

## Demo Notebook (Colab-style cells)

### 0) Setup: synthesize multi-state sensor data (self-contained)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(7)

# Simulate 3 operating states for a sensor window: Normal / Drift / Noisy
n = 450
n_per = n // 3

# Features per window (engineer these like in Class 3/4):
# mean_v, std_v, slope, p2p (peak-to-peak), energy (sum of squares proxy)
normal = np.column_stack([
    np.random.normal(3.20, 0.02, n_per),   # mean
    np.random.normal(0.03, 0.01, n_per),   # std
    np.random.normal(0.00, 0.01, n_per),   # slope
    np.random.normal(0.20, 0.05, n_per),   # peak-to-peak
    np.random.normal(1.00, 0.10, n_per),   # energy
])

drift = np.column_stack([
    np.random.normal(3.32, 0.03, n_per),
    np.random.normal(0.04, 0.01, n_per),
    np.random.normal(0.03, 0.01, n_per),   # positive slope
    np.random.normal(0.25, 0.05, n_per),
    np.random.normal(1.20, 0.12, n_per),
])

noisy = np.column_stack([
    np.random.normal(3.20, 0.04, n_per),
    np.random.normal(0.12, 0.03, n_per),   # high std
    np.random.normal(0.00, 0.02, n_per),
    np.random.normal(0.45, 0.08, n_per),   # large p2p
    np.random.normal(1.60, 0.20, n_per),   # high energy
])

X = np.vstack([normal, drift, noisy])
cols = ["mean_v","std_v","slope","p2p","energy"]
df = pd.DataFrame(X, columns=cols)
df.head()
```

### 1) Standardize, try K values, elbow + silhouette

```python
from collections import OrderedDict

scaler = StandardScaler()
Xs = scaler.fit_transform(df)

inertia = []
sil_scores = []

K = range(2, 8)
for k in K:
    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    labels = km.fit_predict(Xs)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(Xs, labels))

fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(list(K), inertia, marker="o")
ax[0].set_title("Elbow: Inertia vs k"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia"); ax[0].grid(True)

ax[1].plot(list(K), sil_scores, marker="o")
ax[1].set_title("Silhouette vs k"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette"); ax[1].grid(True)
plt.tight_layout(); plt.show()

OrderedDict(zip(K, sil_scores))
```

### 2) Fit K-means with chosen k (e.g., 3), inspect cluster centers

```python
k = 3
kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
labels = kmeans.fit_predict(Xs)
df["cluster"] = labels

# Unscale centers back to feature space for interpretability
centers_std = kmeans.cluster_centers_
centers = pd.DataFrame(scaler.inverse_transform(centers_std), columns=cols)
centers["cluster"] = range(k)
centers
```

### 3) PCA to 2D for visualization

```python
pca = PCA(n_components=2, random_state=0)
Xp = pca.fit_transform(Xs)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative variance:", pca.explained_variance_ratio_.sum())

plt.figure(figsize=(6,5))
for c in range(k):
    idx = (labels == c)
    plt.scatter(Xp[idx,0], Xp[idx,1], s=15, label=f"cluster {c}", alpha=0.8)
plt.title("Clusters in PCA space")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True); plt.legend()
plt.show()
```

### 4) Feature importance (PCA loadings) + center comparison

```python
loadings = pd.DataFrame(pca.components_, columns=cols, index=["PC1","PC2"])
loadings  # how each original feature contributes to PCs

# Compare cluster centers across features
centers.set_index("cluster")
```

### 5) Quick interpretation helper

```python
summary = centers.copy()
summary["interpretation"] = [
    "Higher mean/slope → drift-like?",
    "High std/p2p/energy → noisy regime?",
    "Baseline stats → normal?",
][:k]
summary
```

---

## Classworks (5) — Skeleton Code (students fill in)

### Classwork 1: Standardize + elbow method

```python
# ==========================================
# CLASSWORK 1: SCALE + ELBOW
# ==========================================
# Task:
# 1) Standardize your DataFrame 'df' (columns in list 'cols').
# 2) For k in 2..8, fit KMeans and store inertia.
# 3) Plot inertia vs k; pick a candidate k.

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# scaler = StandardScaler()
# Xs = scaler.???(df[cols])

# inertias = []
# for k in range(2, 9):
#     km = KMeans(n_clusters=???, n_init=20, random_state=0)
#     km.???(Xs)
#     inertias.append(km.???)

# plt.figure()
# plt.plot(range(2,9), inertias, marker="o")
# plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("Elbow")
# plt.grid(True); plt.show()
```

### Classwork 2: Silhouette score sweep

```python
# ==========================================
# CLASSWORK 2: SILHOUETTE SWEEP
# ==========================================
# Task:
# 1) Using Xs, compute silhouette_score for k=2..8.
# 2) Make a small table of k vs silhouette.
# 3) Choose k with reasoning.

from sklearn.metrics import silhouette_score
import pandas as pd

# rows = []
# for k in range(2, 9):
#     km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
#     labels = km.labels_
#     sil = silhouette_score(???, ???)
#     rows.append({"k": k, "silhouette": sil})
# pd.DataFrame(rows)
```

### Classwork 3: PCA 2D projection + colored clusters

```python
# ==========================================
# CLASSWORK 3: PCA VISUALIZATION
# ==========================================
# Task:
# 1) Fit PCA(n_components=2) on Xs.
# 2) Transform Xs -> Xp (2D).
# 3) Fit KMeans with your chosen k; scatter PC1 vs PC2 colored by cluster.

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# pca = PCA(n_components=???, random_state=0)
# Xp = pca.???(Xs)
# print("Explained variance ratio:", pca.???)

# km = KMeans(n_clusters=???, n_init=20, random_state=0).fit(Xs)
# labels = km.labels_

# plt.figure(figsize=(6,5))
# for c in range(???: ???):   # iterate cluster ids
#     idx = (labels == c)
#     plt.scatter(Xp[idx,0], Xp[idx,1], s=15, label=f"cluster {c}")
# plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Clusters in PCA space")
# plt.grid(True); plt.legend(); plt.show()
```

### Classwork 4: Interpret centers (unscale) and name clusters

```python
# ==========================================
# CLASSWORK 4: INTERPRET CLUSTERS
# ==========================================
# Task:
# 1) Take your fitted KMeans centers in standardized space.
# 2) Inverse-transform back to original units (scaler.inverse_transform).
# 3) Put into a DataFrame and add a 'label_name' column with your interpretation.

import pandas as pd

# centers_std = km.cluster_centers_
# centers_orig = scaler.???(centers_std)
# centers_df = pd.DataFrame(centers_orig, columns=cols)
# centers_df["label_name"] = ["???", "???", "???"][:centers_df.shape[0]]
# centers_df
```

### Classwork 5: PCA loadings (feature contributions) + mini report

```python
# ==========================================
# CLASSWORK 5: PCA LOADINGS + REPORT
# ==========================================
# Task:
# 1) Get PCA components_ and create a loadings table with columns=cols and index=["PC1","PC2"].
# 2) Identify top-2 contributing features for each PC.
# 3) Write 2-3 lines explaining how PCs relate to your domain (e.g., std/energy ~ noise axis).

import numpy as np
import pandas as pd

# loadings = pd.DataFrame(pca.???, columns=cols, index=["PC1","PC2"])
# # For each PC, find absolute contribution rankings
# top_pc1 = loadings.loc["PC1"].abs().sort_values(ascending=False).head(2)
# top_pc2 = loadings.loc["PC2"].abs().sort_values(ascending=False).head(2)
# print("Top PC1 features:", top_pc1.index.tolist())
# print("Top PC2 features:", top_pc2.index.tolist())

# # TODO (text cell or print statements): brief interpretation
```

---

### Wrap-up / Homework Challenge

* **Stability check**: rerun K-means with different `random_state` values; do clusters/centers change much? Quantify using **Adjusted Rand Index** after refitting, or compare center distances.
* **Practical twist**: add a rare “faulty” regime (e.g., 5% of samples with extreme std/energy). Can K-means isolate it? If not, try **k=4** or mark the smallest, farthest cluster as *anomaly*. Write a 4-line note: what would your team do next (collect more data? change features? try Isolation Forest in Class 13)?

