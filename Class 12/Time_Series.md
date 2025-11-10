# Class 12: Time-Series Analysis, FFT, Feature Extraction

**Lab:** FFT on audio signals with SciPy + project update (research ideas)

---

### Overview

Time-series data (sensor logs, ECG, audio) hides patterns across both **time** and **frequency**. With the **Fast Fourier Transform (FFT)**, we decompose signals into sine waves, revealing dominant frequencies, noise bands, and harmonics. This session teaches you how to:

* Handle time-domain data (sampling rate, duration).
* Compute FFT using NumPy/SciPy.
* Extract spectral features (peak freq, band energy).
* Apply to real audio (WAV) samples.
* Connect features to research ideas (fault detection, speech recognition, predictive maintenance).

---

## Lecture Notes (Key Concepts)

* **Sampling Rate (fs):** how many samples per second; Nyquist theorem → max detectable frequency = fs/2.
* **FFT:** converts signal from time → frequency domain.

  * Input: N samples (time).
  * Output: N/2 frequencies (magnitude + phase).
* **Spectral Features:**

  * Peak frequency (Hz)
  * Spectral centroid (“center of mass” of frequencies)
  * Band energy (sum over ranges)
* **Windowing:** FFT assumes periodicity → use windows (Hamming, Hann) to reduce leakage.
* **Practical uses:** fault vibration detection, power harmonics analysis, speech/audio recognition, biomedical signals.

---

## Demo Notebook (Colab-style cells)

### 0) Generate synthetic signals (sine + noise)

```python
import numpy as np
import matplotlib.pyplot as plt

fs = 1000   # Sampling rate (Hz)
T = 1.0     # Duration (seconds)
t = np.linspace(0, T, int(fs*T), endpoint=False)

# Signal: sine at 50 Hz + 120 Hz + noise
sig = 0.7*np.sin(2*np.pi*50*t) + 0.3*np.sin(2*np.pi*120*t) + 0.1*np.random.randn(len(t))

plt.figure(figsize=(8,3))
plt.plot(t[:200], sig[:200])
plt.title("Time-Domain Signal (first 200 samples)")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.grid(True); plt.show()
```

### 1) FFT with NumPy

```python
from scipy.fft import fft, fftfreq

N = len(sig)
yf = fft(sig)             # FFT complex spectrum
xf = fftfreq(N, 1/fs)     # Frequency bins

# Only positive freqs (one-sided spectrum)
pos_mask = xf >= 0
xf_pos = xf[pos_mask]
yf_pos = 2.0/N * np.abs(yf[pos_mask])

plt.figure(figsize=(8,3))
plt.plot(xf_pos, yf_pos)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
plt.xlim(0, 250)
plt.grid(True); plt.show()
```

### 2) FFT on audio file (WAV)

```python
import scipy.io.wavfile as wav

fs_audio, data = wav.read("sample_audio.wav")  # provide your own WAV file
print("Sample rate:", fs_audio, "Hz | Shape:", data.shape)

# If stereo, take one channel
if data.ndim > 1:
    data = data[:,0]

# 1-second slice
slice_data = data[:fs_audio]
t_audio = np.linspace(0, 1, fs_audio, endpoint=False)

# FFT
yf_aud = fft(slice_data)
xf_aud = fftfreq(len(slice_data), 1/fs_audio)
mask = xf_aud >= 0

plt.figure(figsize=(8,3))
plt.plot(xf_aud[mask], np.abs(yf_aud[mask])/len(slice_data))
plt.title("Audio Spectrum (first 1s)")
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
plt.xlim(0, 4000)
plt.grid(True); plt.show()
```

### 3) Feature extraction (spectral centroid, peak freq)

```python
mag = np.abs(yf_aud[mask])
freqs = xf_aud[mask]

# Peak frequency
peak_idx = np.argmax(mag)
peak_freq = freqs[peak_idx]
print("Peak Frequency:", round(peak_freq,2), "Hz")

# Spectral centroid
centroid = np.sum(freqs*mag)/np.sum(mag)
print("Spectral Centroid:", round(centroid,2), "Hz")
```

### 4) Band energy feature (e.g., 0–300 Hz, 300–1000 Hz)

```python
band1 = (freqs >= 0) & (freqs < 300)
band2 = (freqs >= 300) & (freqs < 1000)

energy_band1 = np.sum(mag[band1]**2)
energy_band2 = np.sum(mag[band2]**2)

print("Band1 energy (0–300Hz):", round(energy_band1,2))
print("Band2 energy (300–1000Hz):", round(energy_band2,2))
```

### 5) Research project link

```markdown
# Project Brainstorm
- Fault detection: use vibration FFT peaks to classify bearing wear.
- Speech recognition: FFT → MFCCs as features for models.
- Power systems: detect harmonic distortion (3rd, 5th harmonics).
- Biomedical: ECG → FFT for arrhythmia features.
```

---

## Classworks (5) — Skeleton Code (students fill in)

### Classwork 1: Synthetic FFT

```python
# ==========================================
# CLASSWORK 1: SYNTHETIC FFT
# ==========================================
# Task:
# 1) Create signal with two sine waves (e.g., 60 Hz, 150 Hz) + noise.
# 2) Plot time-domain (first 200 samples).
# 3) Compute FFT and plot frequency spectrum up to 300 Hz.

fs = ???     # sampling rate
T  = ???     # duration
t  = np.linspace(0, T, int(fs*T), endpoint=False)

# sig = ??? * np.sin(2*np.pi*60*t) + ??? * np.sin(2*np.pi*150*t) + noise
# yf = fft(sig); xf = fftfreq(len(sig), 1/fs)
# mask = xf >= 0
# plt.plot(???, ???)
# plt.xlim(0, 300); plt.show()
```

### Classwork 2: Audio FFT

```python
# ==========================================
# CLASSWORK 2: AUDIO FFT
# ==========================================
# Task:
# 1) Load 'my_audio.wav'.
# 2) Extract one channel, take 2s slice.
# 3) FFT and plot amplitude spectrum up to 5kHz.

import scipy.io.wavfile as wav

# fs, data = wav.read("my_audio.wav")
# if data.ndim > 1: data = data[:,0]
# slice_data = data[:2*fs]
# yf = fft(slice_data); xf = fftfreq(len(slice_data), 1/fs)
# mask = xf >= 0
# plt.plot(???, ???); plt.xlim(0, 5000); plt.show()
```

### Classwork 3: Spectral centroid

```python
# ==========================================
# CLASSWORK 3: SPECTRAL CENTROID
# ==========================================
# Task:
# 1) Using FFT magnitude, compute spectral centroid.
# 2) Print in Hz.
# (Formula: sum(freq * mag) / sum(mag))

# mag = np.abs(yf[mask])
# freqs = xf[mask]
# centroid = np.sum(???) / np.sum(???)
# print("Spectral Centroid:", centroid, "Hz")
```

### Classwork 4: Band energy

```python
# ==========================================
# CLASSWORK 4: BAND ENERGY
# ==========================================
# Task:
# 1) Compute energy in two bands: [0–500Hz], [500–2000Hz].
# 2) Print both values.

# band1 = (freqs >= 0) & (freqs < 500)
# band2 = (freqs >= 500) & (freqs < 2000)
# E1 = np.sum(mag[band1]**2)
# E2 = np.sum(mag[band2]**2)
# print("Band1:", E1, " | Band2:", E2)
```

### Classwork 5: FFT windowing

```python
# ==========================================
# CLASSWORK 5: WINDOWING EFFECT
# ==========================================
# Task:
# 1) Create a sine signal of 55Hz, length=1s.
# 2) Compute FFT with and without Hann window.
# 3) Compare leakage in plots.

from scipy.signal import hann

# sig = np.sin(2*np.pi*55*t)
# sig_win = sig * hann(len(sig))
# fft_raw = np.abs(fft(sig))
# fft_win = np.abs(fft(sig_win))
# plt.plot(xf[mask], fft_raw[mask], label="No window")
# plt.plot(xf[mask], fft_win[mask], label="Hann window")
# plt.legend(); plt.show()
```

---

## Project Update Prompt (Research Ideas)

* **Pick 1 dataset type**: vibration, power harmonics, audio, biomedical.
* Run FFT, compute 2–3 spectral features.
* Write 3–4 lines: how could these features be used in your planned project (fault detection, classification, anomaly detection, etc.)?

---

## Wrap-up / Homework Challenge

* Extend Classwork 3–4: compute **spectral roll-off** (frequency below which 85% of total energy lies).
* Write a mini “Research Idea Note” in your notebook connecting FFT features to your project domain.
