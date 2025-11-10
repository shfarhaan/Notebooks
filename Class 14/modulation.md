# Class 14: Communication Systems — Modulation Classification & Clustering

**Lab:** Classify signal modulations
**Project Update:** results + paper outline

---

### Overview

Modulation = how info is encoded onto a carrier wave (amplitude, frequency, phase, or combos). Detecting modulation is crucial for cognitive radios, spectrum monitoring, and SDR-based IoT.

We’ll simulate simple modulated signals, extract features (I/Q samples, spectrogram stats), and train classifiers. For unsupervised, we’ll cluster signals based on spectral shape.

---

### Lecture Notes (Key Ideas)

* **I/Q Representation**: real (in-phase) + imaginary (quadrature) components.
* **Common modulations**: AM, FM, PSK, QAM.
* **Classification pipeline**:

  1. Generate/collect signal samples.
  2. Preprocess (normalize, maybe FFT or spectrogram).
  3. Extract features (statistical or deep embeddings).
  4. Train classifier (SVM, RF, CNN for raw IQ).
* **Unsupervised clustering**: cluster unlabeled signals (K-means/PCA).
* **Evaluation**: Accuracy, confusion matrix for classification; silhouette score for clustering.

---

## Demo Notebook (Colab-style)

### 0) Generate synthetic modulated signals (simplified)

```python
import numpy as np, matplotlib.pyplot as plt
np.random.seed(14)

fs = 1000   # samples/sec
T = 1.0     # duration (1s)
t = np.linspace(0, T, int(fs*T), endpoint=False)
fc = 100    # carrier frequency

# Messages
msg = np.sign(np.sin(2*np.pi*3*t))  # square wave baseband

# AM (Amplitude Modulation)
am = (1 + 0.5*msg) * np.sin(2*np.pi*fc*t)

# FM (Frequency Modulation)
kf = 50
fm = np.sin(2*np.pi*fc*t + kf*np.cumsum(msg)/fs)

# BPSK (Binary Phase Shift Keying)
bpsk = np.sin(2*np.pi*fc*t + np.pi*(msg>0))

signals = {"AM":am, "FM":fm, "BPSK":bpsk}

fig, axes = plt.subplots(3,1,figsize=(8,6))
for ax,(name,s) in zip(axes,signals.items()):
    ax.plot(t[:200], s[:200])
    ax.set_title(f"{name} signal (first 200 samples)")
plt.tight_layout(); plt.show()
```

### 1) Extract simple features (time & freq domain)

```python
from scipy.fft import fft, fftfreq

def extract_features(sig):
    N = len(sig)
    yf = np.abs(fft(sig)[:N//2])
    xf = fftfreq(N,1/fs)[:N//2]
    # Simple features: spectral centroid, max freq, energy
    centroid = np.sum(xf*yf)/np.sum(yf)
    maxf = xf[np.argmax(yf)]
    energy = np.sum(yf**2)
    return [centroid,maxf,energy]

X, y = [], []
for label,(name,sig) in enumerate(signals.items()):
    for _ in range(50): # generate noisy copies
        noise = np.random.normal(0,0.3,len(sig))
        feat = extract_features(sig+noise)
        X.append(feat); y.append(name)

import pandas as pd
df = pd.DataFrame(X, columns=["centroid","maxf","energy"])
df["label"] = y
df.head()
```

### 2) Train/test split + classifier

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

X_train,X_test,y_train,y_test = train_test_split(df[["centroid","maxf","energy"]], df["label"], test_size=0.3, random_state=0)
clf = RandomForestClassifier(n_estimators=200, random_state=0)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))

ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred), display_labels=clf.classes_).plot(cmap="Blues")
plt.show()
```

### 3) Clustering without labels

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

Xs = StandardScaler().fit_transform(df[["centroid","maxf","energy"]])
Xp = PCA(n_components=2).fit_transform(Xs)

km = KMeans(n_clusters=3, n_init=20, random_state=0).fit(Xs)
print("Silhouette score:", silhouette_score(Xs, km.labels_))

plt.figure()
for c in range(3):
    plt.scatter(Xp[km.labels_==c,0], Xp[km.labels_==c,1], s=20, label=f"cluster {c}")
plt.title("KMeans Clusters in PCA space")
plt.legend(); plt.grid(True); plt.show()
```

### 4) Visualization: features per modulation

```python
import seaborn as sns
sns.pairplot(df, hue="label", vars=["centroid","maxf","energy"])
plt.suptitle("Feature Distributions per Modulation", y=1.02)
plt.show()
```

---

## Classworks (5) — Skeleton Code

### Classwork 1: Generate another modulation (QPSK)

```python
# ==========================================
# CLASSWORK 1: GENERATE QPSK
# ==========================================
# Task:
# 1) Create a QPSK signal using four phases (0, π/2, π, 3π/2).
# 2) Add noise and plot first 200 samples.
# 3) Append to signals dict as "QPSK".

# msg_bits = np.random.choice([0,1,2,3], size=len(t))
# qpsk = np.sin(2*np.pi*fc*t + (np.pi/2)*msg_bits)
# signals["QPSK"] = ???
# plt.plot(t[:200], ???[:200])
# plt.title("QPSK (noisy, first 200 samples)")
# plt.show()
```

### Classwork 2: Feature extraction

```python
# ==========================================
# CLASSWORK 2: FEATURE EXTRACTION
# ==========================================
# Task:
# 1) Write a function extract_features(sig) that computes:
#    - Spectral centroid
#    - Peak frequency
#    - Total energy
# 2) Test on AM and FM signals.

# def extract_features(sig):
#     N = len(sig)
#     yf = np.abs(fft(sig)[:N//2])
#     xf = fftfreq(N,1/fs)[:N//2]
#     centroid = ???
#     maxf = ???
#     energy = ???
#     return [centroid,maxf,energy]

# print("AM features:", extract_features(signals["AM"]))
# print("FM features:", extract_features(signals["FM"]))
```

### Classwork 3: Classification with RF

```python
# ==========================================
# CLASSWORK 3: RF CLASSIFIER
# ==========================================
# Task:
# 1) Build feature DataFrame for AM, FM, BPSK.
# 2) Train RandomForestClassifier.
# 3) Print accuracy and confusion matrix.

# df = pd.DataFrame([...], columns=["centroid","maxf","energy"])
# df["label"] = [...]
# X_train,X_test,y_train,y_test = train_test_split(???, ???, test_size=0.3)
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(???, ???)
# y_pred = clf.predict(???)
# print("Accuracy:", ???.score(???, ???))
# ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test).plot()
# plt.show()
```

### Classwork 4: KMeans clustering

```python
# ==========================================
# CLASSWORK 4: KMEANS CLUSTERING
# ==========================================
# Task:
# 1) Normalize features using StandardScaler.
# 2) Apply KMeans with k=3.
# 3) Plot PCA 2D projection colored by cluster.

# scaler = StandardScaler()
# Xs = scaler.fit_transform(df[["centroid","maxf","energy"]])
# pca = PCA(n_components=2).fit(Xs)
# Xp = pca.transform(Xs)
# km = KMeans(n_clusters=3, random_state=0).fit(Xs)
# plt.scatter(Xp[:,0], Xp[:,1], c=km.labels_)
# plt.show()
```

### Classwork 5: Research extension

```markdown
# ==========================================
# CLASSWORK 5: PAPER OUTLINE PREP
# ==========================================
# Task:
# 1) Write a short paper outline (Markdown):
#    - Problem: why modulation classification matters.
#    - Methods: dataset, features, classifiers.
#    - Results: metrics, plots.
#    - Discussion: strengths, limitations.
#    - Conclusion: summary + future work.
```

---

## Lab: Classify Signal Modulations

1. Generate 3–4 modulation types (AM, FM, BPSK, QPSK).
2. Extract time/freq-domain features.
3. Train a classifier (RF, SVM, or simple NN).
4. Evaluate with confusion matrix + metrics.
5. Write results into your **paper draft (Class 10–11 template)**.

---

## Project Update: Results + Paper Outline

* Add your **classification results** table + confusion matrix to your draft.
* Draft a **Results section** with a figure (accuracy, per-class metrics).
* Expand your **Discussion** with insights: which modulations are confused and why? What features might help (spectrogram, CNN)?

---

### Wrap-up / Homework Challenge

* Extend lab with **clustering only** (ignore labels). Can clusters line up with true modulations?
* Try deep feature extraction: compute spectrograms (using `scipy.signal.spectrogram`) and feed them to a CNN (optional advanced step).

---
