# Class 4: Data Visualization with Matplotlib (Signals, Noise & Insight)

### Overview

Plots are microscopes for data. With a few careful visuals, you can spot sensor drift, noise bursts, clipping, and timing bugs faster than scrolling numbers. We’ll build clean plots, annotate events, and compare raw vs. cleaned signals.

---

### Lecture Notes (fast + focused)

* **Line plots**: time-series sanity checks (raw, cleaned, rolling mean).
* **Subplots**: compare multiple signals/scales in one figure.
* **Histograms**: distribution of noise/values; quick outlier sniff test.
* **Annotations**: mark spikes, thresholds, or events with arrows/labels.
* **Plot as debugger**: always plot before training ML—catch NaNs, misaligned time, flatlines.

---

## Demo Notebook (Colab-style cells)

### 0) Setup: make a tiny dataset (self-contained)

```python
import numpy as np
import pandas as pd

np.random.seed(7)
t = np.arange(0, 100, 1)  # 0..99 seconds
# Base signal ~3.2V with small noise; inject a spike around t=40
v_raw = 3.2 + 0.05*np.random.randn(len(t))
v_raw[40] = 4.8  # spike
df = pd.DataFrame({"time_s": t, "voltage_v": v_raw})

# Basic cleaning
df["voltage_v_filled"] = df["voltage_v"].interpolate(limit_direction="both")
df["voltage_v_clipped"] = df["voltage_v_filled"].clip(3.0, 3.6)
df["roll_mean_5"] = df["voltage_v_clipped"].rolling(5, min_periods=1).mean()

df.head()
```

### 1) Single line plot (raw vs. cleaned)

```python
import matplotlib.pyplot as plt

plt.figure()
plt.plot(df["time_s"], df["voltage_v"], marker=".", label="raw")
plt.plot(df["time_s"], df["voltage_v_clipped"], marker=".", label="cleaned")
plt.title("Voltage vs Time: Raw vs Cleaned")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.legend()
plt.show()
```

### 2) Subplots: raw, cleaned, rolling mean (stacked)

```python
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

axes[0].plot(df["time_s"], df["voltage_v"], marker=".")
axes[0].set_title("Raw Signal")
axes[0].set_ylabel("V")
axes[0].grid(True)

axes[1].plot(df["time_s"], df["voltage_v_clipped"], marker=".")
axes[1].set_title("Cleaned (clipped)")
axes[1].set_ylabel("V")
axes[1].grid(True)

axes[2].plot(df["time_s"], df["roll_mean_5"], marker=".")
axes[2].set_title("Rolling Mean (window=5)")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("V")
axes[2].grid(True)

fig.tight_layout()
plt.show()
```

### 3) Histogram: distribution of voltages (raw vs cleaned)

```python
plt.figure()
plt.hist(df["voltage_v"], bins=20, alpha=0.6, label="raw")
plt.hist(df["voltage_v_clipped"], bins=20, alpha=0.6, label="cleaned")
plt.title("Voltage Distribution")
plt.xlabel("Voltage (V)")
plt.ylabel("Count")
plt.legend()
plt.show()
```

### 4) Annotate spike and threshold line

```python
# Find spike index (largest deviation from median)
idx_spike = np.argmax(np.abs(df["voltage_v"] - df["voltage_v"].median()))
t_spike = df.loc[idx_spike, "time_s"]
v_spike = df.loc[idx_spike, "voltage_v"]

plt.figure()
plt.plot(df["time_s"], df["voltage_v"], marker=".", label="raw")
plt.axhline(3.6, linestyle="--", label="upper threshold (3.6V)")
plt.scatter([t_spike],[v_spike], s=80, label="detected spike")
plt.title("Spike Annotation + Threshold")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.legend()

# Annotate with arrow
plt.annotate(
    "Spike here",
    xy=(t_spike, v_spike),
    xytext=(t_spike+8, v_spike-0.6),
    arrowprops=dict(arrowstyle="->", lw=1),
)
plt.show()
```

### 5) Compare two sensors + secondary axis (when needed)

```python
# Synthetic second channel with slight offset and drift
df["voltage_ch2"] = df["voltage_v_clipped"] + 0.07*np.sin(2*np.pi*df["time_s"]/30) + 0.02

fig, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(df["time_s"], df["voltage_v_clipped"], label="ch1 (cleaned)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Ch1 Voltage (V)")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(df["time_s"], df["voltage_ch2"], label="ch2", linestyle="--")
ax2.set_ylabel("Ch2 Voltage (V)")

# Simple merged legend
lns = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc="lower right")
plt.title("Two Channels (twin axes)")
plt.show()
```

---

## Classworks (5) — Skeleton Code (students fill in)

### Classwork 1: Clean time-series plot

```python
# ==========================================
# CLASSWORK 1: CLEAN TIME-SERIES PLOT
# ==========================================
# Task:
# 1) Load a CSV 'class4_ts.csv' with columns time_s, voltage_v. (Create if missing.)
# 2) Plot voltage vs time with markers.
# 3) Add title, axis labels, grid, and legend.
# 4) Save the figure as 'class4_ts.png'.

import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("class4_ts.csv")   # or create it with pd.DataFrame(...).to_csv(...)
# plt.figure()
# plt.plot(???, ???, marker=".", label="voltage")
# plt.title(???)
# plt.xlabel(???)
# plt.ylabel(???)
# plt.grid(???)
# plt.legend()
# plt.savefig("class4_ts.png", dpi=150)
# plt.show()
```

### Classwork 2: Subplots for raw vs. cleaned vs. rolling

```python
# ======================================================
# CLASSWORK 2: SUBPLOTS (RAW vs CLEANED vs ROLLING)
# ======================================================
# Task:
# 1) Starting from df with 'voltage_v', create cleaned 'voltage_clean' by clipping [3.0, 3.6].
# 2) Add rolling mean 'roll_mean_7' (window=7, min_periods=1).
# 3) Create a 3-row subplot figure showing raw, cleaned, and rolling.
# 4) Share x-axis, label y-axes, set titles, and call tight_layout().

import numpy as np
import matplotlib.pyplot as plt

# df["voltage_clean"] = df["voltage_v"].???(3.0, 3.6)
# df["roll_mean_7"] = df["voltage_clean"].rolling(window=???, min_periods=1).???
# fig, axes = plt.subplots(???, 1, figsize=(8,8), sharex=True)
# axes[0].plot(df["time_s"], df["voltage_v"], marker=".")
# axes[0].set_title("Raw")
# axes[1].plot(df["time_s"], df["voltage_clean"], marker=".")
# axes[1].set_title("Cleaned")
# axes[2].plot(df["time_s"], df["roll_mean_7"], marker=".")
# axes[2].set_title("Rolling Mean (7)")
# axes[2].set_xlabel("Time (s)")
# for ax in axes: 
#     ax.set_ylabel("V"); ax.grid(True)
# fig.???()
# plt.show()
```

### Classwork 3: Histogram + vertical reference lines

```python
# ======================================================
# CLASSWORK 3: HISTOGRAM + REFERENCE LINES
# ======================================================
# Task:
# 1) Plot a histogram of 'voltage_v' with 25 bins.
# 2) Add vertical lines for mean and +/- one std.
# 3) Add legend and labels.

import numpy as np
import matplotlib.pyplot as plt

# mu = ???   # mean
# sigma = ???  # std
# plt.figure()
# plt.hist(df["voltage_v"], bins=???, alpha=0.7, label="raw")
# plt.axvline(mu, linestyle="--", label="mean")
# plt.axvline(mu + sigma, linestyle=":")
# plt.axvline(mu - sigma, linestyle=":")
# plt.title("Voltage Distribution")
# plt.xlabel("V")
# plt.ylabel("Count")
# plt.legend()
# plt.show()
```

### Classwork 4: Annotate anomalies

```python
# ==========================================
# CLASSWORK 4: ANNOTATE ANOMALIES
# ==========================================
# Task:
# 1) Detect indices where 'voltage_v' > 3.6 or < 3.0 (choose bounds).
# 2) Plot the time series and scatter the anomalies.
# 3) Annotate the highest anomaly with an arrow.

import numpy as np
import matplotlib.pyplot as plt

# mask_hi = df["voltage_v"] > ???
# mask_lo = df["voltage_v"] < ???
# idxs = np.where(mask_hi | mask_lo)[0]
# plt.figure()
# plt.plot(df["time_s"], df["voltage_v"], marker=".", label="raw")
# plt.scatter(df["time_s"].iloc[idxs], df["voltage_v"].iloc[idxs], s=60, label="anomalies")
# # Pick the most extreme anomaly:
# idx_max = idxs[???]  # choose via np.argmax on abs deviation if you want
# plt.annotate("Anomaly", xy=(df["time_s"].iloc[idx_max], df["voltage_v"].iloc[idx_max]),
#              xytext=(df["time_s"].iloc[idx_max]+5, df["voltage_v"].iloc[idx_max]),
#              arrowprops=dict(arrowstyle="->", lw=1))
# plt.title("Annotated Anomalies")
# plt.xlabel("Time (s)")
# plt.ylabel("V")
# plt.grid(True)
# plt.legend()
# plt.show()
```

### Classwork 5: Two-channel comparison (shared x, twin y)

```python
# ======================================================
# CLASSWORK 5: TWO-CHANNEL COMPARISON
# ======================================================
# Task:
# 1) Assume df has 'voltage_ch1' and 'voltage_ch2'. If not, create ch2 by offsetting ch1.
# 2) Plot ch1 on primary axis; ch2 on a twin y-axis (ax.twinx()).
# 3) Add a combined legend and proper labels.

import matplotlib.pyplot as plt

# if "voltage_ch1" not in df.columns:
#     df["voltage_ch1"] = df["voltage_v"]
# if "voltage_ch2" not in df.columns:
#     df["voltage_ch2"] = df["voltage_ch1"] + 0.05

# fig, ax1 = plt.subplots(figsize=(8,4))
# l1 = ax1.plot(df["time_s"], df["voltage_ch1"], label="ch1")
# ax1.set_xlabel("Time (s)")
# ax1.set_ylabel("Ch1 (V)")
# ax1.grid(True)
# ax2 = ax1.twinx()
# l2 = ax2.plot(df["time_s"], df["voltage_ch2"], linestyle="--", label="ch2")
# ax2.set_ylabel("Ch2 (V)")
# lines = l1 + l2
# labels = [ln.get_label() for ln in lines]
# ax1.legend(lines, labels, loc="lower right")
# plt.title("Two Channels (Twin Axes)")
# plt.show()
```

---

### Wrap-up / Homework Challenge

* Recreate **Classwork 2** but add a **fourth subplot**: rolling **std** (window=7) below the rolling mean. Discuss how std helps spot noisy intervals.
* Bonus: Export figures with informative filenames (include date/time or parameters), and write a 3–4 line lab note interpreting what you see (where/when is data unstable? did cleaning help?).

---
