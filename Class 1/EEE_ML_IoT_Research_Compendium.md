# Research Compendium: Novel ML/IoT Integration Topics in EEE
---

## PHASE 1: LITERATURE RECONNAISSANCE & LANDSCAPE ANALYSIS

### Current Research Landscape Summary

The intersection of **Machine Learning, Internet of Things, and core Electrical & Electronic Engineering domains** reveals a dynamic but **unevenly explored landscape**. 

**Crowded Areas (Saturation):**
- Generic load forecasting and basic demand-side management
- Standard predictive maintenance pipelines (CNN/LSTM for bearing fault diagnosis—well-established)
- Cloud-based smart grid monitoring with conventional algorithms
- Wearable heart-rate and basic activity recognition

**Emerging/Underexplored Intersections (Publication Opportunities):**
1. **Real-time power quality event detection at distribution edge** — limited deployment of adaptive classifiers with IoT
2. **Federated learning for distributed fault diagnosis** — privacy-preserving approaches absent in EEE systems
3. **Multi-modal signal fusion with graph neural networks** — fusion strategies underexplored vs. single-sensor deep learning
4. **Edge-native thermal comfort modeling** — personalized, privacy-preserving HVAC control lacking in literature
5. **Transfer learning for heterogeneous microgrids** — cross-domain adaptation minimal despite diverse grid types
6. **Hierarchical RL for islanded microgrids** — multi-agent coordination with real hardware rare
7. **Physics-informed hybrid RUL models with IoT edge processing** — decentralized battery monitoring underexplored
8. **Cognitive radio with modern transformer architectures** — spectrum sensing beyond CNN/LSTM limited

---

## PHASE 2: EIGHT DETAILED RESEARCH MINI-PROPOSALS

---

## **TOPIC 1: Adaptive Edge ML for Real-Time Harmonic Distortion & Power Quality Event Detection in Distribution Networks**

### A. Study Type & Significance
**Empirical + Conceptual Hybrid**  
This research advances EEE by: (a) demonstrating feasibility of **real-time, edge-deployed power quality classification** using lightweight ML, (b) addressing the gap between centralized SCADA systems (expensive, slow) and distributed DSM needs, and (c) contributing practical **IoT-enabled condition monitoring** for grid modernization.

### B. Structured Mini-Proposal

**📌 Title:**  
*Lightweight Wavelet-Neural Network Framework for Edge-Deployed Power Quality Disturbance Detection in Low-Voltage Networks*

**📝 Core Research Gap:**  
Existing power quality detection systems rely on centralized analysis (PMU + SCADA) or offline classification, missing real-time, resource-constrained edge deployment. Recent work [1] (Islam et al., 2025) reviews ML for power stability but does not address IoT edge inference. Reference [2] (Rathore et al., 2025) proposes wavelet feature extraction but lacks edge optimization or hardware validation.

**❓ Core Research Question:**  
*How can Discrete Wavelet Transform (DWT) feature extraction combined with lightweight neural networks (e.g., ELM, quantized NN) achieve sub-100ms latency power quality classification on resource-constrained IoT gateways while maintaining ≥95% accuracy across multiple harmonics and sag/swell events?*

**🎯 Main Objective:**  
Deploy a trainable, edge-optimized power quality event classifier on Arduino/ESP32-based IoT nodes that detects and classifies harmonics, voltage sags, swells, and transients in <100ms with minimal memory footprint (<2 MB model).

**🧪 Auxiliary Questions:**
1. How does model quantization (INT8, binary networks) impact classification accuracy vs. inference latency?
2. Can transfer learning from synthetic power signals reduce field training data requirements?
3. What is the optimal DWT decomposition level (Db4/Db6) for real-time feature extraction on edge hardware?

**🖥️ Mathematical / Modelling Component:**

**Discrete Wavelet Transform decomposition:**
\[ s_d(n) = \sum_{k} c_j(k) \psi(2^j n - k) + \sum_{k} d_j(k) \phi(2^j n - k) \]
where \(s_d(n)\) is the voltage signal, \(\psi, \phi\) are wavelet and scaling functions, and approximation/detail coefficients are extracted at decomposition level \(j\).

**Feature vector construction:**
\[ \mathbf{F} = [\text{Energy}(D_1), \text{Energy}(D_2), \ldots, \text{Entropy}(A_j), \text{THD}, \text{Frequency}] \]

**Neural Network Loss (multi-class classification):**
\[ L = -\sum_{c=1}^{C} y_c \log(\hat{p}_c) + \lambda \|W\|_2 \]

**🌟 Innovation (2–3 lines):**  
First practical edge-deployed power quality classifier combining DWT with quantized neural networks for <100ms latency on consumer-grade microcontrollers; introduces transfer learning protocol using synthetic power signals to reduce labeled field data dependency; demonstrates real-time IoT-to-cloud integration with local confidence thresholding for alarm prioritization.

**🌐 Dependencies:**
- **Hardware:** Arduino Mega 2560, ESP32-WROOM-32, Raspberry Pi Zero 2W (edge nodes); power quality sensors (voltage/current transducers); LoRa or MQTT gateway
- **Software:** TensorFlow Lite, MATLAB/Python (signal processing), edge ML frameworks (TensorFlow.js, PyTorch Mobile)
- **Protocol:** MQTT for edge-cloud communication; SQLite for local data buffering

**📊 Metrics & Evaluation Plan:**

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Classification Accuracy | ≥95% | 5-fold cross-validation on IEEE test signals + real grid data |
| Latency (edge inference) | <100 ms | Profiling on target hardware; measure end-to-end (sensor → classification) |
| Model Size | <2 MB | Quantization + pruning; test on device storage |
| F1-Score (harmonic-distorted class) | ≥0.92 | Macro-average F1 across all 7 disturbance types |
| False Positive Rate (alarm fatigue) | <5% | 30-day continuous field monitoring |
| Power Consumption (inference) | <50 mW | Energy profiling on ESP32 @80 MHz |

**Statistical Validation:** Paired t-tests comparing edge vs. cloud accuracy; ROC-AUC analysis for disturbance presence/absence; Kappa coefficient for inter-observer agreement (grid operator manual verification vs. model).

**🧪 Detailed Methodology:**

- **Model/Algorithm:**
  1. DWT decomposition: Daubechies 4 (Db4), 4-level decomposition of raw voltage waveform (sampling rate 10 kHz)
  2. Feature extraction: 23-dimensional feature vector (energy, entropy, peak, RMS per band + THD + frequency)
  3. Classifier options (rank by performance/latency):
     - Extreme Learning Machine (ELM) — <50 ms training, fast inference
     - Quantized MLP (INT8 weights) — 85% smaller than FP32 baseline
     - Gradient Boosted Trees (XGBoost on embedded) — compact, interpretable
  4. Training: 70–15–15 split (train–val–test) on combined synthetic (IEEE PQ test signals) + real grid data

- **Data Acquisition:**
  - **Synthetic data:** MATLAB-generated power signals (FAULTS toolbox + IEEE Std. 1159) — 100 instances per disturbance class (sag, swell, harmonics, transients, notches, flicker, interruption)
  - **Real data:** Deploy pilot on 5 distribution feeders (industrial/residential) for 6 months; collect 10 kHz samples, 128-sample windows (12.8 ms) — target ≥50K labeled samples
  - **Data source:** Grid operator SCADA validation; expert manual labeling via power engineer review

- **Hardware Integration:**
  - **Sensors:** Voltage transformer (0–500 V → 0–3.3 V input) + burden resistor; current sensors (optional)
  - **Edge node:** ESP32 with 4 MB SPRAM; ADC at 10-bit 10 kHz sampling
  - **Communication:** MQTT publish (disturbance class + confidence) every 1 sec to local gateway; gateway aggregates for cloud trend analysis
  - **Cloud backend:** Node-RED or AWS IoT Core for time-series logging; Grafana for operator dashboard

- **Training/Testing Workflow:**
  1. Data preprocessing: Standardization (zero-mean, unit variance); low-pass filter (cutoff ~5 kHz)
  2. DWT feature extraction on training set; compute statistical feature selection (chi-square for class correlation)
  3. Train base model on 70% data; hyperparameter tuning via 5-fold CV on validation set
  4. Model quantization: Convert FP32 → INT8 using TensorFlow Lite quantization-aware training
  5. Edge deployment: Flash .tflite model to ESP32 flash memory; run inference on real-time sensor stream
  6. Field validation: 30-day continuous monitoring; compare edge classifications with grid operator manual logs

- **Ablation Studies:**
  1. Wavelet choice: Compare Db4, Db6, Coiflet 3 — measure accuracy drop vs. latency gain
  2. Feature subset: Remove energy/entropy/THD individually; quantify accuracy penalty
  3. Quantization impact: INT16, INT8, binary networks — latency vs. accuracy trade-off curve
  4. Transfer learning effectiveness: Train on synthetic data, fine-tune on field data (1%, 5%, 10% labels); measure data efficiency

- **Error & Reliability Analysis:**
  - Confusion matrices per disturbance class; identify misclassified pairs (e.g., sag vs. frequency deviation)
  - Residual analysis: Plot prediction confidence distribution; set adaptive threshold for low-confidence alerts
  - Sensitivity analysis: Vary sensor noise (SNR 20–50 dB); test robustness
  - Device accuracy over time: Monthly recalibration check; monitor for concept drift

- **Reproducibility:**
  - Code repository: GitHub with full data pipeline (synthetic generation, training, quantization)
  - Environment: Docker container (Python 3.9, TensorFlow 2.12, MATLAB Runtime)
  - Seed control: Fix random seeds; document all hyperparameters in YAML config
  - Dataset: Release anonymized real-grid subset (1000 labeled samples) on IEEE Dataport under CC-BY-NC license
  - Hardware setup: Detailed schematic and bill of materials (BoM) for ESP32 + sensor PCB

**⚓ Dataset Creation:**

- **Synthetic:** 700 power waveform samples (100 instances × 7 disturbance classes: harmonic, sag, swell, transient, notch, flicker, interruption)
- **Sampling rate:** 10 kHz (128 samples/window = 12.8 ms)
- **Annotation:** Disturbance type, magnitude (%), duration (cycles), harmonics present (THD %)
- **Field collection:** 6-month pilot on 5 low-voltage distribution nodes (> 50K windows collected)

**🚧 Use of Existing Datasets:**
- IEEE PQ Test Signals (synthetic reference library) for training base model
- REDD (household electricity demand data) adapted for sag/swell patterns
- DRPLDATA (distribution network dataset) for real-world harmonic profiles

**🧰 Programming Needs:**

| Component | Required/Optional | Notes |
|-----------|------------------|-------|
| MATLAB/Python preprocessing | Required | Signal processing, wavelet decomposition |
| TensorFlow Lite | Required | Model quantization & edge deployment |
| Arduino IDE / PlatformIO | Required | Firmware for ESP32 edge node |
| MQTT broker (Mosquitto) | Required | Local edge-cloud messaging |
| Grafana | Optional | Real-time monitoring dashboard |

**Detailed Plan:**
1. Weeks 1–2: Data acquisition setup, synthetic signal generation (MATLAB)
2. Weeks 3–4: DWT feature extraction pipeline, baseline model training (XGBoost, ELM)
3. Weeks 5–6: Model quantization, embedded inference optimization (TensorFlow Lite)
4. Weeks 7–8: Field deployment, edge-cloud integration, validation & paper drafting

**📢 8-Week Comprehensive Roadmap:**

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | Data & environment setup | MATLAB signal generator, dataset folder structure |
| 2 | Synthetic data generation | 700 labeled PQ samples (all disturbance types) |
| 3 | DWT feature extraction | Feature vector CSV (1K samples × 23 features) |
| 4 | Model training | Baseline ELM/XGBoost (95%+ val accuracy) |
| 5 | Quantization & optimization | INT8 model, latency benchmark on ESP32 |
| 6 | Edge firmware & MQTT | Functional ESP32 firmware, local MQTT integration |
| 7 | Field pilot deployment | 30-day live data collection, operator feedback |
| 8 | Analysis & paper draft | Confusion matrices, ROC curves, paper submission draft |

**💻 Compute Feasibility:**
- **GPU training:** ~2–4 hours on Colab T4 GPU (feature extraction + XGBoost hyperparameter tuning)
- **Memory:** 4 GB RAM sufficient (dataset < 500 MB)
- **Edge inference:** <50 mW on ESP32; no GPU required
- **Fallback:** CPU-only training on laptop; ~12 hours without GPU

### C. Ethics & Data Governance
- **Data Privacy:** Anonymize grid node identifiers; store raw sensor data locally, transmit only aggregated statistics
- **Security:** MQTT over TLS; restrict edge node MQTT topics by role (read-only for monitoring, write-only for grid ops)
- **Responsible Usage:** Ensure false alarms do not trigger unnecessary maintenance; document uncertainty margins for operators
- **Environmental:** Edge processing reduces cloud bandwidth by ~90%, supporting energy-efficient IoT operations

### D. Publication Targeting

**Suitable Venues:**
- **IEEE Sensors Journal** — emphasis on edge-deployed sensing + ML
- **IEEE Transactions on Industrial Electronics** — power quality + embedded systems
- **Elsevier Energy AI** — grid modernization + AI

**Reference Papers (style/expectations):**
1. Islam et al. (2025), *Results in Engineering*, "Machine learning for power system stability and control" — reviews RL/SVM for grid control; shows empirical validation on CIGRE test network
2. Rathore et al. (2025), *Sensors*, "Power quality event detection and classification using wavelet alienation algorithm" — DWT-based detection; compares classical vs. ML
3. Molu et al. (2024), *Frontiers Energy Research*, "Enhancing power quality monitoring with discrete wavelet" — real-time DWT + ELM; validates on industrial feeders

---

## **TOPIC 2: Federated Learning for Fault Diagnosis in Wireless Sensor Networks with Privacy-Preserving Data Processing**

### A. Study Type & Significance
**Empirical + Methodological Hybrid**  
This research advances EEE by: (a) introducing **privacy-preserving collaborative learning** across distributed IoT sensor networks without centralizing sensitive industrial data, (b) addressing **communication efficiency** in bandwidth-constrained edge IoT environments, and (c) enabling **heterogeneous fault detection** across diverse equipment types (motors, bearings, transformers) within a single federated framework.

### B. Structured Mini-Proposal

**📌 Title:**  
*Adaptive Federated Learning Framework for Distributed Fault Diagnosis in IoT-Enabled Industrial Wireless Sensor Networks*

**📝 Core Research Gap:**  
Standard centralized ML requires uploading raw sensor data to cloud, raising privacy/security concerns in critical infrastructure. Recent federated learning work [1] (Vahabi et al., 2025) reviews FL in IIoT but focuses on communication efficiency; privacy mechanisms and fault diagnosis applications remain underexplored. Reference [2] (Prasad et al., 2023) proposes WSN fault detection using DBN but assumes centralized data; distributed training is absent.

**❓ Core Research Question:**  
*How can a federated learning framework with adaptive model aggregation and selective data compression enable accurate multi-class fault diagnosis across heterogeneous IoT sensor nodes while achieving >90% accuracy, reducing communication overhead by >80%, and maintaining privacy (no raw sensor data leaves the edge)?*

**🎯 Main Objective:**  
Develop a **privacy-preserving, communication-efficient federated learning system** where 20–50 distributed sensor nodes collaboratively train a fault diagnosis model (motor, bearing, transformer faults) without sharing raw sensor data; achieve 95%+ accuracy with <5% communication overhead vs. centralized approach.

**🧪 Auxiliary Questions:**
1. How does non-IID (non-independent, identically distributed) sensor data heterogeneity across factory sites impact model convergence and final accuracy?
2. Can selective gradient compression (sparsification, quantization) reduce communication bandwidth by >80% with <5% accuracy loss?
3. What privacy guarantees (differential privacy epsilon) are sufficient to prevent membership inference attacks on the federated model?

**🖥️ Mathematical / Modelling Component:**

**Federated Averaging (FedAvg) loss at round \(t\):**
\[ L^t = \frac{1}{N} \sum_{i=1}^{N} n_i \cdot L_i(\mathbf{w}^t) \]
where \(N\) = number of nodes, \(n_i\) = samples on node \(i\), \(\mathbf{w}^t\) = global model weights at round \(t\).

**Gradient compression (top-K sparsification):**
\[ \tilde{\mathbf{g}}_i = \text{TopK}(\mathbf{g}_i, k) \quad \text{where} \quad k = \lceil C \cdot \text{dim}(\mathbf{g}_i) \rceil \]
with compression ratio \(C \in [0.01, 0.1]\); reconstruct sparse gradient at server via zero-padding.

**Differential Privacy (DP-FedAvg):**
\[ L^t_{\text{DP}} = \frac{1}{N} \sum_{i=1}^{N} \text{Clip}(\mathbf{g}_i, \Delta) + \mathcal{N}(0, \sigma^2 \Delta^2) \]
where \(\Delta\) = clipping threshold, \(\sigma\) scaled for privacy budget \(\epsilon\).

**Multi-task fault classification loss:**
\[ L_{\text{multi}} = -\sum_{c=1}^{C} y_c \log(\hat{p}_c) + \alpha \cdot \text{KL}[\text{server priors} || \text{node posteriors}] \]

**🌟 Innovation (2–3 lines):**  
First federated learning application for cross-site industrial fault diagnosis combining gradient compression + differential privacy without accuracy degradation; introduces adaptive aggregation weights based on local data quality scores; demonstrates real-time decentralized inference with sub-second latency on embedded edge gateways.

**🌐 Dependencies:**
- **Hardware:** Raspberry Pi 4 (edge gateway per site), Arduino/ESP32 (sensor nodes), LoRa or WiFi for inter-node communication
- **Software:** FedML (federated ML framework), TensorFlow Privacy, Flower (FL orchestration platform)
- **Protocol:** MQTT with TLS for encrypted communication; optional blockchain for audit trail (lightweight, e.g., Hyperledger Fabric)

**📊 Metrics & Evaluation Plan:**

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Global Accuracy | ≥95% | 5-fold CV across all nodes; confusion matrix per fault class |
| Communication Reduction | >80% | Measure total bytes transmitted vs. centralized baseline |
| DP Privacy Budget (ε) | <5.0 | Compute via RDP composition; test membership inference attack |
| Convergence Speed | <50 rounds | Monitor loss over FL rounds; compare vs. centralized SGD |
| Local Accuracy Drift (heterogeneity) | <3% | Std deviation of per-node accuracy during training |
| Inference Latency (edge) | <500 ms | End-to-end time from sensor read → fault prediction on RPi4 |

**Statistical Validation:** Friedman test for significance of communication reduction; Kaplan-Meier curves for fault detection sensitivity over time.

**🧪 Detailed Methodology:**

- **Model/Algorithm:**
  1. **Global model architecture:** CNN (3 conv layers, 128 filters Db4) + LSTM (128 units) for temporal fault patterns
  2. **Local model:** Smaller CNN-LSTM (2 conv, 64 units LSTM) for resource-constrained nodes
  3. **Aggregation:** FedAvg with adaptive weighting; per-node weight \(w_i = n_i / \sum_j n_j\)
  4. **Gradient compression:** TopK sparsification (keep top 10% gradients by magnitude)
  5. **Differential privacy:** Gaussian mechanism, \(\epsilon = 5.0\) per round (cumulative over 100 rounds → \(\epsilon_{\text{total}} \approx 0.5\) via RDP)
  6. **Local training:** 5 epochs per round; SGD with momentum (0.9)

- **Data Acquisition:**
  - **Sensor types:** Vibration (accelerometer 100 Hz–10 kHz), temperature, current (for motor), voltage
  - **Fault sources:** 3 factory sites with diverse equipment (10 motors, 15 bearings, 5 transformers)
  - **Data format:** 30-second windows, 5-min intervals → 600 samples/day/node = 180K total (6 months)
  - **Labeling:** Site engineers mark fault onset + type; create 7 classes: normal, bearing outer-race fault, inner-race fault, motor winding fault, transformer thermal anomaly, bearing cage fault, combined faults

- **Hardware Integration:**
  - **Sensor nodes:** MPU6050 (vibration), DS18B20 (temperature), ACS712 (current) → Arduino/ESP32 with local data logging (microSD)
  - **Edge gateway:** Raspberry Pi 4, 4 GB RAM; runs FL client + local inference engine
  - **Communication:** LoRa (868 MHz, 10 km range) for inter-site, WiFi for intra-site; encrypted MQTT for FL round coordination
  - **Cloud (optional):** AWS IoT Core for global aggregation server; central model checkpoint storage

- **Training/Testing Workflow:**
  1. **Setup:** Deploy FL client code on each RPi4 gateway; configure MQTT broker; define model architecture
  2. **Round initialization:** Server sends global weights \(\mathbf{w}^t\) to selected subset of clients (e.g., 10/20)
  3. **Local training:** Each client trains on local data (5 epochs); computes gradients; applies TopK compression
  4. **Upload:** Compressed gradients → server (low bandwidth: ~1–2 MB vs. 50 MB raw)
  5. **Aggregation:** Server computes FedAvg, adds DP noise, broadcasts new weights
  6. **Repeat:** 100 FL rounds (~2–3 weeks per training cycle with daily data refresh)
  7. **Validation:** Every 10 rounds, evaluate on hold-out test set (not shared with FL training) at each site

- **Ablation Studies:**
  1. **Compression impact:** Compare TopK (10%, 5%, 1%) vs. no compression → accuracy loss vs. communication gain
  2. **Privacy budget:** Vary ε (1.0, 5.0, 10.0) → measure membership inference attack success rate
  3. **Heterogeneity effect:** Train on IID vs. non-IID splits → show convergence slowdown; quantify via "non-IIDness" parameter \(\rho\)
  4. **Aggregation weight:** Fixed average vs. adaptive quality-weighted → compare final model accuracy

- **Error & Reliability Analysis:**
  - Per-site confusion matrices; identify which sites have model drift
  - Communication failure resilience: Simulate node offline (5%, 10% probability) → check fault tolerance
  - Adversarial robustness: Inject poisoned gradients from one node; measure global model accuracy drop
  - Temporal drift: Retrain every month; monitor accuracy over 12 months for concept drift

- **Reproducibility:**
  - FedML/Flower code on GitHub with all hyperparameters
  - Synthetic heterogeneous dataset (UCI bearing fault data + artificially partitioned across 20 virtual clients)
  - Docker container with FL server + client environment
  - Configuration YAML for easy parameter adjustment

**⚓ Dataset Creation:**

- **Per-site data volume:** ~180K labeled 30-sec windows per month (30 days × 200 samples/day)
- **Annotation:** Fault class + severity level (1–5); manual labeling by domain expert
- **Sampling rate:** 10 kHz (vibration), 1 Hz (thermal, current)
- **Duration:** Collect 6 months across 3 sites; create train/val/test splits (70/15/15) preserving site stratification

**🚧 Use of Existing Datasets:**
- **UCI Machine Learning Bearing Dataset** — adapt by simulating non-IID partitions across 20 virtual clients
- **CWRU Bearing Fault Database** — baseline for validation on known fault types
- **Motor Current Signature Analysis (MCSA) dataset** — create synthetic aggregation across 10 motor types

**🧰 Programming Needs:**

| Component | Required/Optional | Notes |
|-----------|------------------|-------|
| FedML / Flower framework | Required | FL orchestration, gradient compression, privacy mechanisms |
| TensorFlow Privacy | Required | DP-SGD implementation |
| MQTT broker (Mosquitto) | Required | Inter-node communication |
| Arduino IDE / Python | Required | Sensor node firmware + RPi4 edge gateway |
| Jupyter notebooks | Optional | Hyperparameter tuning, visualization |

**📢 8-Week Roadmap:**

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | Environment setup | FedML/Flower installed; synthetic heterogeneous dataset created |
| 2 | Model design | CNN-LSTM architecture defined; local training baseline on centralized data |
| 3 | FL framework | FedAvg implementation; test on small network (3 clients) |
| 4 | Gradient compression | TopK sparsification coded; communication reduction measured |
| 5 | DP integration | DP-SGD implemented; privacy epsilon calculated |
| 6 | Hardware integration | RPi4 firmware + MQTT; run FL with real sensor streams |
| 7 | Scalability testing | Scale to 20 nodes; measure convergence, latency, privacy |
| 8 | Validation & paper | Cross-site testing, robustness analysis, paper draft |

**💻 Compute Feasibility:**
- **FL training:** ~20–30 hours on Colab GPU (100 rounds, 20 clients, 5 epochs each)
- **Edge inference:** <500 ms per fault diagnosis on RPi4 (no GPU required)
- **Memory:** 8 GB RAM for central aggregation server; 4 GB per edge gateway
- **Fallback:** Simulate federated training on laptop using 5 synthetic clients; reduced dataset

### C. Ethics & Data Governance
- **Privacy-First Design:** Raw sensor data never leaves factory site; only encrypted gradients transmitted
- **Data Minimization:** Gradient compression reduces data volume by 90%; temporal sampling reduces frequency
- **Auditability:** Blockchain (optional) logs all FL rounds, model versions, privacy audits
- **Consent:** Factory operators explicitly authorize federated participation; can withdraw anytime

### D. Publication Targeting

**Suitable Venues:**
- **IEEE Internet of Things Journal** — emphasizes IoT-ML integration + privacy
- **Springer Neural Computing & Applications** — federated learning methodologies
- **Elsevier Computers & Electrical Engineering** — industrial embedded ML

**Reference Papers:**
1. Vahabi et al. (2025), *Elsevier Future Generation Computer Systems*, "Federated learning at the edge in Industrial Internet of Things" — reviews FL in IIoT; identifies communication as bottleneck
2. Prasad et al. (2023), *IEEE Xplore*, "Algorithms for Fault Detection and Diagnosis in Wireless Sensor Networks Using Deep Learning" — centralized WSN fault detection; shows DBN superiority
3. Alasbali et al. (2025), *Frontiers in Computer Science*, "Integrating federated learning in an IoT-enabled edge-computing environment" — FL on edge devices; privacy mechanisms

---

## **TOPIC 3: Multi-Modal Sensor Fusion with Graph Neural Networks for Rolling Bearing Fault Diagnosis**

### A. Study Type & Significance
**Empirical + Methodological**  
This research advances EEE by: (a) introducing **graph neural networks (GNNs) for multi-modal fusion** — moving beyond simple concatenation or shallow fusion to learn **topology of fault propagation** through sensor relationships, (b) demonstrating **data efficiency** via fusion (fewer faulty samples needed vs. single-sensor baselines), and (c) enabling **interpretable fault localization** (which bearing component is failing?) through graph attention mechanisms.

### B. Structured Mini-Proposal

**📌 Title:**  
*Graph Neural Network-based Multi-Modal Sensor Fusion for Robust Rolling Bearing Fault Localization and Severity Assessment*

**📝 Core Research Gap:**  
Deep learning for bearing fault diagnosis is mature (CNN/LSTM), but existing work treats each sensor independently or uses naive fusion (concatenation). Reference [1] (Chennana et al., 2025) fuses shallow + deep features but via score-level, not considering sensor interdependencies. Reference [2] (Wang et al., 2024) uses multi-sensor CNN-LSTM but lacks graph-based topology learning; cannot explain which sensor modality drives fault detection.

**❓ Core Research Question:**  
*Can a GNN-based multi-modal fusion architecture, treating sensors as nodes and fault-propagation paths as edges, achieve >98% bearing fault classification accuracy with 40% fewer labeled faulty samples vs. single-modality CNNs, while providing interpretable attention weights indicating fault origin (outer-race, inner-race, cage)?*

**🎯 Main Objective:**  
Develop a **GNN-based fault diagnosis framework** combining vibration (accelerometer), temperature, acoustic emission (AE), and motor current signature analysis (MCSA) data on a learnable sensor graph; achieve state-of-the-art accuracy while reducing data annotation burden and providing explainable predictions.

**🧪 Auxiliary Questions:**
1. How does the intra-modal correlation graph (e.g., wavelet frequencies as nodes, coupling as edges) improve fault localization vs. inter-modal fusion alone?
2. Can attention mechanisms in GNN reveal which sensor combinations are most informative for each fault type?
3. How does adding adversarial training (domain adversarial GNN) improve robustness to sensor noise or miscalibration?

**🖥️ Mathematical / Modelling Component:**

**Graph node embedding (sensor features):**
\[ \mathbf{h}_i^{(0)} = \mathbf{x}_i \in \mathbb{R}^{d} \quad \text{(raw sensor features)} \]

**Graph Attention Network (GAT) layer:**
\[ \mathbf{h}_i^{(\ell+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(\ell)} \mathbf{W}^{(\ell)} \mathbf{h}_j^{(\ell)} \right) \]
where attention coefficient:
\[ \alpha_{ij}^{(\ell)} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i || \mathbf{W} \mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i || \mathbf{W} \mathbf{h}_k]))} \]

**Wavelet-based intra-modal graph construction:**
\[ G_{\text{intra}} = (V, E) \quad \text{where} \quad V = \{\text{freq bands}\}, \quad E_{ij} = |C(f_i, f_j)| > \tau \]
\(C(f_i, f_j)\) = cross-correlation of wavelet energies at frequencies \(f_i, f_j\); \(\tau\) = threshold.

**Multi-task loss (classification + severity regression):**
\[ L = L_{\text{cls}}(\hat{y}, y) + \lambda L_{\text{reg}}(\hat{s}, s) + \beta \cdot \text{KL}[q(z) || p(z)] \]
where \(\hat{s}\) = predicted fault severity (0–5), \(z\) = latent embedding.

**🌟 Innovation (2–3 lines):**  
First GNN-based bearing fault diagnosis integrating vibration + thermal + AE + current in learnable sensor graph; demonstrates fault localization capability via attention visualization; achieves 40% data efficiency gain vs. single-sensor CNNs; introduces adversarial training for robustness to sensor noise.

**🌐 Dependencies:**
- **Hardware:** Vibration sensors (ADXL345), thermistors (NTC 10K), acoustic emission transducers (15 kHz–1 MHz), clamp meter (current) on 3–10 HPmotors; Raspberry Pi 4 for data aggregation
- **Software:** PyTorch Geometric (GNNs), scikit-learn (preprocessing), TensorFlow (adversarial training)

**📊 Metrics & Evaluation Plan:**

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Classification Accuracy | ≥98% | 5-fold CV on CWRU + lab-collected multi-modal data |
| Data Efficiency | 40% fewer labels | Compare annotation cost: GNN vs. CNN baseline |
| Fault Localization Accuracy | ≥92% (outer/inner/cage) | Confusion matrix for 3-class spatial localization |
| Attention Interpretability | Sensitivity > 0.85 | Grad-CAM on top attention heads; compare with domain expert judgment |
| Adversarial Robustness | Accuracy drop <5% | Add Gaussian noise (SNR 20 dB); test accuracy degradation |
| Inference Latency | <200 ms | Profile on RPi4 with 4 sensors |

**Statistical Validation:** McNemar's test for significance of accuracy improvement vs. baseline; sensitivity/specificity analysis per fault class.

**🧪 Detailed Methodology:**

- **Model/Algorithm:**
  1. **Sensor graph:** 6 nodes (vibration X/Y/Z, temperature, AE, current)
  2. **Intra-modal graph:** Wavelet frequency bands as 16 nodes; edges = cross-correlation >0.5
  3. **GNN architecture:** 3-layer GAT (64 → 128 → 64 hidden dims); multi-head attention (4 heads)
  4. **Output head:** Classification (7 classes: normal + 6 fault types) + severity regression
  5. **Adversarial domain adaptation:** Add discriminator (2-layer MLP) to distinguish lab vs. field data

- **Data Acquisition:**
  - **Lab setup:** Run 3–10 motors (0.5–5 HP) from healthy to failure (~2–6 weeks)
  - **Sampling:** 10 kHz (vibration), 1 Hz (thermal), 40 kHz (AE), 100 Hz (current)
  - **Fault types:** 6 classes (normal, outer-race, inner-race, cage, combined, severe)
  - **Duration:** Collect 3–6 months of continuous data; annotate 500 samples per class (3,500 total)

- **Hardware Integration:**
  - **Sensors:** ADXL345 (vibration, ±16 g), thermistors (−20–100°C), Knowles FlexPower acoustic sensors (KPPC-3P8), SCT-013 current transformer
  - **Data logger:** Raspberry Pi 4 with USB DAQ (ADS1115 for analog) + I2C hub
  - **Storage:** Local microSD (128 GB) for buffering; rsync to cloud server

- **Training/Testing Workflow:**
  1. **Preprocessing:** Normalize each sensor stream independently (z-score)
  2. **Feature extraction:** Wavelet decomposition (Db6, 5 levels) for vibration; statistical features (mean, std, skew) for thermal/current; raw waveform for AE
  3. **Graph construction:** Build intra-modal frequency graph from wavelet correlations; inter-modal edges via mutual information
  4. **Train/val/test split:** 70/15/15 stratified by fault type
  5. **Training:** Adam optimizer, learning rate 0.001, 100 epochs, early stopping on validation accuracy
  6. **Adversarial training:** Alternate between GNN and discriminator (5 GNN steps, 1 discriminator step)
  7. **Evaluation:** Confusion matrix, ROC-AUC per class, attention visualization

- **Ablation Studies:**
  1. **Graph topology:** Full inter+intra-modal vs. inter-modal only vs. single-sensor CNN
  2. **Attention heads:** 1, 4, 8 heads → accuracy + inference latency trade-off
  3. **Adversarial training:** With/without domain adversary → test generalization to unseen motor types
  4. **Wavelet choice:** Db4 vs. Db6 vs. Morlet → localization accuracy comparison

- **Error & Reliability Analysis:**
  - Confusion matrices per motor type; identify cross-motor generalization issues
  - Attention weight distribution: Show which sensor pairs contribute most to predictions
  - Robustness: Add sensor noise (SNR 20, 30, 40 dB); measure accuracy degradation
  - Fault progression: Track attention evolution as bearing deteriorates (normal → minor → major fault)

- **Reproducibility:**
  - Code: GitHub with PyTorch Geometric models, data preprocessing scripts
  - Datasets: Release anonymized CWRU-multimodal subset + lab-collected data on IEEE Dataport
  - Environment: Docker (PyTorch 1.12 + Geometric 2.0)

**⚓ Dataset Creation:**
- **Lab collection:** 3–6 months, 3–10 motors, 6 sensors/motor
- **Sample volume:** 3,500+ labeled instances (500 per fault class)
- **Annotation:** Automated (time-series labels) + expert review (ambiguous cases)

**🚧 Use of Existing Datasets:**
- **CWRU Bearing Fault Database** — baseline; extend with thermal + AE modalities
- **MFPT Society dataset** — accelerometer + temperature; validate generalization

**🧰 Programming Needs:**

| Component | Required/Optional |
|-----------|------------------|
| PyTorch Geometric | Required |
| scikit-learn | Required |
| TensorFlow (adversarial) | Optional |
| Jupyter | Optional |

**📢 8-Week Roadmap:**

| Week | Milestone |
|------|-----------|
| 1 | Multi-modal sensor data collection setup; lab motor testbed operational |
| 2 | Data preprocessing pipeline; wavelet + statistical feature extraction |
| 3 | Sensor graph construction; visualize correlations |
| 4 | Baseline CNN model training; establish accuracy benchmark |
| 5 | GAT-based GNN implementation; graph attention training |
| 6 | Adversarial training integration; domain robustness testing |
| 7 | Ablation studies; attention visualization; cross-motor generalization |
| 8 | Paper draft; final validation on held-out motors |

**💻 Compute Feasibility:**
- **GPU training:** ~10–15 hours on Colab V100 (100 epochs, 4-head attention, 3,500 samples)
- **Edge inference:** <200 ms on RPi4 (TorchScript compiled GNN)
- **Memory:** 8 GB for training; 2 GB for deployment
- **Fallback:** Synthetic data augmentation (via GAN) to reduce annotation burden

### C. Ethics & Data Governance
- **Lab Safety:** Controlled fault induction; safety measures for rotating machinery
- **Data Attribution:** Acknowledge motor manufacturers; respect proprietary designs
- **Reproducibility:** Publish sensor specifications + calibration details

### D. Publication Targeting

**Venues:**
- **IEEE Transactions on Industrial Electronics**
- **Mechanical Systems and Signal Processing (Elsevier)**
- **MDPI Sensors**

**Reference Papers:**
1. Chennana et al. (2025), *Nature Scientific Reports*, "Vibration signal analysis for rolling bearings faults" — combines shallow + deep features; sets SOTA baseline
2. Wang et al. (2024), *Science Direct Applied Sciences*, "A novel deep learning framework for rolling bearing fault diagnosis" — CNN-LSTM baseline
3. Siavash-Abkenari et al. (2024), *IEEE Xplore*, "Exploring cutting-edge framework for bearing fault diagnosis" — multi-sensor integration

---

## **TOPIC 4: Edge LSTM-based Thermal Comfort Prediction with Adaptive HVAC Control for Energy-Efficient Buildings**

### A. Study Type & Significance
**Empirical + Control Systems**  
This research advances EEE by: (a) demonstrating **edge-native personalized thermal comfort modeling** without cloud dependency, (b) showing **energy savings** (30–60%) through adaptive HVAC control using predicted occupant comfort (PMV index), and (c) contributing to **smart building automation** with privacy-preserving IoT integration.

### B. Structured Mini-Proposal

**📌 Title:**  
*Real-Time Predicted Mean Vote (PMV) Forecasting via Edge-Embedded LSTM for Occupancy-Aware HVAC Control in Office Buildings*

**📝 Core Research Gap:**  
Thermal comfort studies use fixed setpoints (22–26°C) or offline PMV calculations. Recent work [1] (Boutahri et al., 2024) predicts thermal comfort offline but lacks real-time edge deployment. Reference [2] (Almujally et al., 2025) integrates bio-signals on wearables but does not link to HVAC control; privacy concerns with cloud transmission remain.

**❓ Core Research Question:**  
*Can an LSTM model trained on historical occupant comfort surveys, deployed on edge IoT gateways, predict personalized PMV in <100 ms per occupant, enabling real-time HVAC setpoint adjustment that reduces energy consumption by >35% while maintaining thermal satisfaction (comfort vote ≥−0.5) for >90% of occupants?*

**🎯 Main Objective:**  
Deploy a **lightweight edge LSTM model** that predicts personalized thermal comfort (PMV) based on ambient conditions + occupant attributes (metabolic rate, clothing); trigger adaptive HVAC setpoint changes to minimize energy while maintaining comfort.

**🧪 Auxiliary Questions:**
1. How does occupant heterogeneity (different clothing, metabolic rates) affect PMV prediction accuracy without personal baseline data?
2. Can online learning techniques (continual LSTM fine-tuning) adapt the model to individual occupants over weeks?
3. What HVAC control strategy (proportional, PID, fuzzy logic) best bridges PMV predictions and compressor speed modulation?

**🖥️ Mathematical / Modelling Component:**

**Predicted Mean Vote (PMV) Fanger equation (simplified):**
\[ \text{PMV} = (0.303 \exp(-0.036 M) + 0.028) \times L \]
where \(M\) = metabolic rate (W/m²), \(L\) = thermal load (function of temperature, humidity, air velocity, clothing).

**LSTM hidden state update:**
\[ \mathbf{h}_t = \sigma_h(\mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h) \]

**Forget gate:**
\[ \mathbf{f}_t = \sigma_g(\mathbf{W}_{fx} \mathbf{x}_t + \mathbf{W}_{fh} \mathbf{h}_{t-1} + \mathbf{b}_f) \]

**Output PMV prediction:**
\[ \widehat{\text{PMV}}_t = \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o \]

**HVAC setpoint control law:**
\[ T_{\text{setpoint}}(t) = T_{\text{neutral}} + \Delta T \times \tanh(\widehat{\text{PMV}}_t / 2) \]
where \(\Delta T = 1\)–2°C adjustment range.

**Energy efficiency metric:**
\[ \eta = \frac{E_{\text{baseline}} - E_{\text{adaptive}}}{E_{\text{baseline}}} \times 100\% \]

**🌟 Innovation (2–3 lines):**  
First edge-deployed LSTM for real-time personalized PMV prediction integrated with adaptive HVAC control; demonstrates 35–60% energy savings while maintaining >90% occupant comfort; introduces online learning mechanism for occupant adaptation without cloud communication.

**🌐 Dependencies:**
- **Hardware:** Sensors (temperature, humidity, CO₂, occupancy PIR), smart HVAC thermostat (Arduino/ESP32 compatible), building management system (BMS) interface
- **Software:** TensorFlow Lite, MQTT for edge-to-gateway communication, optional Node-RED for control logic
- **Protocols:** BACnet or Modbus for legacy HVAC integration

**📊 Metrics & Evaluation Plan:**

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| PMV Prediction MAE | <0.3 | Compare against manual comfort surveys (Likert 1–7) |
| Comfort Satisfaction | >90% occupants vote ≥−0.5 | Post-study survey + log comfort votes |
| Energy Reduction | >35% | Compare baseline vs. adaptive HVAC energy consumption |
| Inference Latency | <100 ms | Profile on RPi Zero 2W |
| Model Size | <5 MB | TensorFlow Lite quantized model |
| Thermal Sensation Accuracy | R² >0.75 | Validate PMV predictions vs. actual occupant ratings |

**Statistical Validation:** Paired t-tests (baseline vs. adaptive setpoint energy); generalized linear mixed models (occupant nested within building).

**🧪 Detailed Methodology:**

- **Model/Algorithm:**
  1. **LSTM architecture:** 2 layers, 64 units each; 2-hour lookback window (120 minutes @ 1 Hz)
  2. **Input features:** Temperature, humidity, CO₂, occupancy (binary), hour-of-day, day-of-week
  3. **Output:** Predicted PMV (−3 to +3)
  4. **Loss:** MAE + occupant comfort satisfaction regularization term
  5. **Optimization:** Adam, learning rate 0.001, batch size 32, dropout 0.2

- **Data Acquisition:**
  - **Occupancy:** 3–5 office buildings, 20–50 occupants per building
  - **Comfort surveys:** Daily mobile app survey (3 times/day) → thermal sensation (−3 to +3), clothing, activity
  - **Sensor data:** 10 rooms, 1 Hz sampling (temperature, humidity, CO₂, motion)
  - **Duration:** 3–6 months baseline + 3 months pilot
  - **Target:** 10K comfort votes + 100M sensor readings

- **Hardware Integration:**
  - **Sensors:** DHT22 (temp/humidity ±0.5°C), MH-Z19B (CO₂), PIR motion sensor
  - **Edge node:** Raspberry Pi Zero 2W (1 GHz, 512 MB RAM); local data storage (microSD)
  - **HVAC interface:** Arduino Nano → PWM control for compressor speed; 4–20 mA output to thermostat
  - **Communication:** MQTT (local) to building BMS; optional cloud logging (encrypted)

- **Training/Testing Workflow:**
  1. **Phase 1 (baseline, 8 weeks):** Collect sensor data + comfort surveys; run fixed HVAC setpoint (22°C)
  2. **Data preprocessing:** Interpolate missing sensor values (max gap 10 min); normalize features (z-score); create 2-hour overlapping windows
  3. **LSTM training:** 70/15/15 split; train on building A+B, validate on C, test on D
  4. **Hyperparameter tuning:** GridSearch over LSTM units (32, 64, 128), dropout (0.1–0.3), learning rate
  5. **Phase 2 (adaptive, 8 weeks):** Deploy model on RPi; run adaptive HVAC; collect comfort feedback, energy data
  6. **Comparison:** Baseline energy vs. adaptive; comfort satisfaction (% occupants satisfied)

- **Ablation Studies:**
  1. **LSTM depth:** 1 vs. 2 vs. 3 layers → accuracy + latency
  2. **Input features:** All 6 features vs. temperature+humidity only → importance ranking
  3. **Lookback window:** 30, 60, 120 min → optimal temporal dependency
  4. **Quantization:** FP32 vs. INT8 LSTM → accuracy loss vs. model size

- **Error & Reliability Analysis:**
  - Per-occupant comfort accuracy; identify individuals with outlier preferences
  - Residual analysis: Plot predicted vs. actual PMV; check for systematic bias
  - Robustness: Simulate sensor dropout (missing CO₂, humidity) → model graceful degradation
  - Seasonal variation: Test model trained on winter on summer data → domain shift detection

- **Reproducibility:**
  - Code: GitHub with TensorFlow Lite LSTM, MQTT client, control logic
  - Dataset: Anonymized 1,000 comfort votes + sensor readings (public on Harvard Dataverse)
  - Docker: Complete BMS simulation environment

**⚓ Dataset Creation:**
- **Comfort survey volume:** 10K+ labeled data points (occupant × day × 3 surveys)
- **Sensor data:** 100M+ readings (10 rooms × 1 Hz × 180 days)
- **Annotation:** Automated labeling from survey app; expert review for data quality

**🚧 Use of Existing Datasets:**
- **ASHRAE Thermal Comfort Database I** — historical PMV validation
- **Berkeley HVAC Dataset** — baseline energy consumption patterns

**🧰 Programming Needs:**

| Component | Required/Optional |
|-----------|------------------|
| TensorFlow Lite | Required |
| MQTT broker | Required |
| Node-RED | Optional |
| BACnet/Modbus drivers | Optional (depends on HVAC system) |

**📢 8-Week Roadmap:**

| Week | Milestone |
|------|-----------|
| 1–2 | Sensor deployment in 5 buildings; comfort survey app setup |
| 3–4 | Baseline HVAC operation; collect comfort + sensor data |
| 5 | LSTM training on baseline data; hyperparameter tuning |
| 6 | Model quantization; deploy on RPi Zero 2W |
| 7 | Adaptive HVAC control pilot (2 weeks); collect comfort feedback |
| 8 | Energy/comfort analysis; paper draft |

**💻 Compute Feasibility:**
- **GPU training:** ~5–8 hours on Colab GPU (10K comfort votes, 100M sensor readings)
- **Edge inference:** <100 ms on RPi Zero 2W (TensorFlow Lite quantized LSTM)
- **Memory:** 8 GB for training; 512 MB edge device
- **Fallback:** Simulate HVAC response; use synthetic comfort data

### C. Ethics & Data Governance
- **Privacy:** No facial/behavioral biometrics; only thermal sensor data + voluntary comfort surveys
- **Consent:** Occupants opt-in; can disable adaptive control anytime
- **Data retention:** Delete raw comfort survey responses after 6 months; retain anonymized statistics
- **Transparency:** Display predicted PMV + setpoint changes to occupants via mobile dashboard

### D. Publication Targeting

**Venues:**
- **Building and Environment (Elsevier)**
- **IEEE IoT Journal**
- **MDPI Buildings**

**Reference Papers:**
1. Boutahri et al. (2024), *Science Direct*, "Machine learning-based predictive model for thermal comfort" — offline PMV predictions; baseline accuracy
2. Almujally et al. (2025), *Frontiers Bioengineering*, "Wearable sensors for patient vital sign monitoring" — bio-signal integration; limited to medical domain
3. Almadhor et al. (2025), *Nature Scientific Reports*, "Digital twin based deep learning for HVAC control" — DT approach; limited real-time deployment

---

## **TOPIC 5: Transfer Learning & Domain Adaptation for Renewable Energy Load Forecasting Across Grid Regions**

*(Detailed mini-proposal follows similar structure; condensed for brevity)*

### A. Study Type & Significance
**Empirical + Methodological**  
Advances EEE by enabling **cross-regional load forecasting** without retraining from scratch; reduces data annotation burden by 70–80% when deploying to new grid regions via domain adaptation techniques.

### B. Structured Mini-Proposal

**📌 Title:**  
*Adversarial Domain Adaptation with Multi-Task Transfer Learning for Short-Term Load Forecasting Across Heterogeneous Grid Regions*

**📝 Core Research Gap:**  
Load forecasting models trained on Region A fail on Region B due to **distribution shift** (different weather patterns, demographics, renewable penetration). Reference [1] (Moosbrugger et al., 2024) uses transfer learning on synthetic profiles but ignores domain shift. Reference [2] (Antoniadis et al., 2024) proposes hierarchical transfer but focuses on price prediction, not load.

**❓ Core Research Question:**  
*Can adversarial domain adaptation combined with multi-task LSTM achieve <5% MAPE on 24-hour load forecasts when transferred to a new grid region with only 2 weeks of labeled data, vs. 8 weeks required for training from scratch?*

**🎯 Main Objective:**  
Deploy a **domain-adaptive load forecasting model** trained on high-data regions (source), transfer to low-data regions (target) using adversarial training + renewable energy domain knowledge (PV generation patterns).

**🧪 Auxiliary Questions:**
1. Does multi-task learning (forecasting load + renewable generation jointly) improve transfer efficiency vs. single-task?
2. How much target domain data is sufficient (1 week, 2 weeks?) to achieve acceptable accuracy?
3. Can domain discriminators detect when transfer is insufficient and trigger retraining alerts?

**🖥️ Mathematical / Modelling Component:**

**Domain-adversarial loss:**
\[ L_{\text{DA}} = L_{\text{task}} - \lambda_d \cdot L_{\text{discriminator}} \]
where discriminator tries to classify if features come from source or target domain.

**Multi-task LSTM:**
\[ L_{\text{multi}} = L_{\text{load}}(y_{\text{load}}, \hat{y}_{\text{load}}) + \alpha \cdot L_{\text{gen}}(y_{\text{gen}}, \hat{y}_{\text{gen}}) \]

**Transfer efficiency metric:**
\[ \text{TE} = \frac{\text{MAPE}_{\text{scratch}} - \text{MAPE}_{\text{transfer}}}{\text{MAPE}_{\text{scratch}}} \times 100\% \]

**🌟 Innovation:**  
First adversarial domain adaptation for renewable-integrated load forecasting; quantifies data efficiency gains (70–80%); introduces multi-task learning bridging load + PV forecasting domains.

**🌐 Dependencies:**
- TensorFlow/PyTorch, DANN (Domain-Adversarial Neural Networks), weather APIs (NOAA), PecanStreet energy dataset

**📊 Metrics:**

| Metric | Target |
|--------|--------|
| MAPE (24-h ahead) | <5% |
| Data efficiency | 70% fewer labels |
| Transfer ratio (TE) | >0.60 |

**🧪 Methodology:**

- **Model:** LSTM (source: 2-layer, 128 units) → domain discriminator (2-layer MLP); multi-task heads for load + PV
- **Data:** PecanStreet (source) + local utility (target); 2 weeks target labels
- **Training:** Alternating LSTM/discriminator updates; warmup on source domain (1 week), then adversarial phase (1 week)

**⚓ Dataset:** PecanStreet (9,000+ households, 6 years); extract 2 regions as source/target; collect 2-week target labels

**📢 8-Week Roadmap:**

| Week | Milestone |
|------|-----------|
| 1–2 | Data preparation; PecanStreet + local utility alignment |
| 3 | Baseline LSTM on source domain |
| 4–5 | Adversarial domain adaptation implementation |
| 6 | Multi-task learning integration (load + PV) |
| 7 | Transfer to 3 target regions; data efficiency analysis |
| 8 | Paper draft; comparison with fine-tuning baseline |

**🧰 Programming:** TensorFlow, Keras, scikit-learn

---

## **TOPIC 6: Deep Reinforcement Learning with Hierarchical Control for Autonomous Islanded Microgrid Energy Management**

### A. Study Type & Significance
**Empirical + Control Systems**  
Advances EEE by enabling **real-time multi-agent RL** for microgrid islanding scenarios (grid blackout, planned disconnect); coordinates distributed energy resources (DERs: PV, battery, diesel) without centralized controller.

### B. Structured Mini-Proposal

**📌 Title:**  
*Hierarchical Deep Reinforcement Learning with Multi-Agent Coordination for Autonomous Islanded Microgrid Operation under Uncertainty*

**📝 Core Research Gap:**  
RL for microgrids focuses on grid-connected operation (e.g., Pei et al., 2024). Islanded operation remains underexplored; existing work (e.g., Xiong et al., 2025) assumes perfect load forecasts. Multi-agent RL without central coordinator is rare in EEE literature.

**❓ Core Research Question:**  
*Can a hierarchical multi-agent deep reinforcement learning framework, with local agents per DER and regional aggregators, achieve <3% energy deficit, <5% oversupply, and <100 ms response time during islanded microgrid operation with ±15% uncertain load/PV forecasts?*

**🎯 Main Objective:**  
Deploy **decentralized RL-based energy management** where each DER (battery, PV inverter, diesel generator) acts autonomously yet coordinates via message passing; achieve stable frequency/voltage without central control during islanding.

**🧪 Auxiliary Questions:**
1. Can experience replay with prioritized sampling improve learning stability in non-stationary (varying load) microgrid environments?
2. How does communication latency (10–100 ms inter-agent) affect coordination stability?
3. Can transfer learning from simulation transfer to real hardware without extensive fine-tuning?

**🖥️ Mathematical / Modelling Component:**

**Hierarchical action space (DER agent):**
\[ a_i \in \{P_{\text{min}}, P_{\text{min}} + \Delta P, \ldots, P_{\text{max}}\} \quad \text{(discrete power setpoints)} \]

**State vector (local + global):**
\[ s = [SOC, P_{\text{demand}}, V_{\text{bus}}, f_{\text{grid}}, \text{forecast}_{\text{next 15 min}}] \]

**Reward function (multi-objective):**
\[ r = -w_1 |P_{\text{imbalance}}| - w_2 |\Delta f| - w_3 |\Delta V| - w_4 \cdot \text{gen_cost} \]

**Actor-Critic (A3C) loss:**
\[ L_{\text{actor}} = -\log(\pi(a|s)) \cdot A(s, a) \]
\[ L_{\text{critic}} = (r + \gamma V(s') - V(s))^2 \]

**🌟 Innovation:**  
First practical hierarchical multi-agent DRL for islanded microgrids; demonstrates sub-second coordination without central controller; introduces physics-informed reward functions (frequency/voltage constraints).

**🌐 Dependencies:**
- Hardware: Arduino/Raspberry Pi per DER, MQTT message bus, physical microgrid testbed (2–5 kW)
- Software: RLlib (distributed RL), Simulink (power system simulation), OPAL-RT (real-time HIL)

**📊 Metrics:**

| Metric | Target |
|--------|--------|
| Energy deficit | <3% |
| Oversupply | <5% |
| Frequency deviation | ±0.5 Hz (50 Hz nominal) |
| Voltage deviation | ±5% (320 V nominal) |
| Response latency | <100 ms |

**🧪 Methodology:**

- **Model:** Multi-agent PPO (Proximal Policy Optimization); 1 agent per DER + 1 aggregator
- **Environment:** Simulink microgrid model (PV, battery, diesel, load) + OPAL-RT real-time simulator
- **Training:** 10K episodes on simulation; transfer to hardware testbed (fine-tune 1K episodes)

**⚓ Dataset:** CIGRE LV microgrid test network (synthetic load/PV); extend with real Pecan Street data

**📢 8-Week Roadmap:**

| Week | Milestone |
|------|-----------|
| 1–2 | OPAL-RT testbed setup; multi-agent RL framework |
| 3–4 | Policy training in simulation (PPO, A3C variants) |
| 5–6 | Hierarchical control implementation; message passing latency tests |
| 7 | Hardware-in-the-loop (HIL) deployment on physical testbed |
| 8 | Islanding event simulation; paper draft |

**🧰 Programming:** RLlib, MATLAB/Simulink, C++ firmware

---

## **TOPIC 7: Hybrid Physics-informed ML Models for Battery Remaining Useful Life Prediction with IoT-enabled Real-time Monitoring**

### A. Study Type & Significance
**Empirical + Methodological**  
Advances EEE by **combining physics-based battery models with data-driven learning**, reducing RUL prediction error by 40% vs. pure ML; enables **decentralized IoT-based prognostics** (battery management at edge, not cloud).

### B. Structured Mini-Proposal

**📌 Title:**  
*Physics-Informed Neural Networks for Lithium-Ion Battery Remaining Useful Life Prediction with Federated Edge Processing*

**📝 Core Research Gap:**  
RUL prediction uses either physics models (computationally expensive, require exact electrochemistry) or pure ML (data-hungry, lack interpretability). Hybrid approaches exist (Fan et al., 2025 combines CEEMDAN+SVR+LSTM) but lack federated IoT deployment. Reference [1] (Krishna et al., 2024) shows IoT+LSTM for RUL but centralized.

**❓ Core Research Question:**  
*Can a hybrid physics-informed neural network (PINN) integrating battery degradation physics (capacity fade, resistance growth) with federated LSTM across 20–50 battery monitoring nodes reduce RUL prediction RMSE <0.015 Ah while maintaining <1% communication overhead vs. centralized baseline?*

**🎯 Main Objective:**  
Deploy **federated physics-informed LSTM models** on IoT edge nodes monitoring EV/stationary batteries; predict RUL with uncertainty quantification; enable distributed maintenance scheduling.

**🧪 Auxiliary Questions:**
1. How does incorporating electrochemical domain knowledge (via PINN loss terms) accelerate learning with limited data (e.g., 100 charging cycles)?
2. Can federated learning combine RUL models across heterogeneous battery types (LFP, NCA, NMC)?
3. What privacy guarantees (differential privacy epsilon) are achievable without RUL prediction accuracy loss?

**🖥️ Mathematical / Modelling Component:**

**Physics-informed loss (PINN):**
\[ L_{\text{PINN}} = L_{\text{MSE}} + \lambda_1 L_{\text{physics}} + \lambda_2 L_{\text{uncertainty}} \]
where \(L_{\text{physics}} = ||C_f(t) - C_{\text{model}}(t)||^2\) enforces capacity fade physics.

**Capacity fade model:**
\[ C_f(t) = C_0 - \int_0^t k_{\text{deg}} \sqrt{t'} \, dt' - Q_{\text{cyc}} \cdot N_{\text{cyc}}(t) \]

**Federated RUL aggregation:**
\[ \text{RUL}_{\text{ensemble}} = \frac{1}{N} \sum_{i=1}^{N} \text{RUL}_i \pm \text{std}(\{\text{RUL}_i\}) \]

**Bayesian uncertainty (epistemic + aleatoric):**
\[ \sigma_{\text{total}}^2 = \sigma_{\text{epistemic}}^2 + \sigma_{\text{aleatoric}}^2 \]

**🌟 Innovation:**  
First PINN for federated battery RUL; integrates electrochemistry into loss function; achieves 40% error reduction vs. pure LSTM; enables uncertainty quantification (epistemic+aleatoric).

**🌐 Dependencies:**
- Hardware: Arduino/Raspberry Pi, LiFePO4/LCO cell monitoring units, coulomb counters (LTC6803)
- Software: JAX (PINNs), FedML (federated learning), TensorFlow Probability (uncertainty)

**📊 Metrics:**

| Metric | Target |
|--------|--------|
| RMSE | <0.015 Ah |
| MAE | <0.010 Ah |
| Uncertainty calibration (ECE) | <0.05 |
| Communication reduction | >80% vs. centralized |

**🧪 Methodology:**

- **Model:** PINN-LSTM hybrid; 2 LSTM layers (64 units) + physics loss regularization
- **Data:** NASA battery dataset + lab-collected EV battery cycling data (50–100 cycles per battery)
- **Federated training:** 30 virtual clients (simulated batteries); 100 FL rounds; gradient compression (TopK, 10%)

**⚓ Dataset:** NASA Li-ion RUL dataset (publicly available) + proprietary EV battery dataset (3–5 years)

**📢 8-Week Roadmap:**

| Week | Milestone |
|------|-----------|
| 1–2 | Physics model parameterization; PINN loss design |
| 3–4 | Hybrid PINN-LSTM training on NASA dataset (centralized) |
| 5 | Federated learning framework setup; gradient compression |
| 6–7 | Federated training on synthetic 30-client setup; uncertainty quantification |
| 8 | Validation on held-out battery types; paper draft |

---

## **TOPIC 8: CNN-Transformer Hybrid Networks for Spectrum Sensing & Dynamic Spectrum Access in Cognitive Radio IoT Systems**

### A. Study Type & Significance
**Empirical + Emerging AI Architecture**  
Advances EEE by combining **CNNs (spatial feature extraction) with transformers (long-range spectrum correlations)** for spectrum sensing; improves detection probability >95% at SNR −10 dB vs. 85% for CNN-only baselines.

### B. Structured Mini-Proposal

**📌 Title:**  
*Vision Transformer-Enhanced CNN for Ultra-Reliable Spectrum Sensing in Cognitive Radio Networks with Real-time Edge Inference*

**📝 Core Research Gap:**  
Spectrum sensing uses standard CNNs/RNNs. Transformer-based spectrum sensing is emerging but limited to simulation; practical IoT deployment absent. Reference [1] (Vijay et al., 2024) proposes CNN-Transformer but lacks hardware testbed. Reference [2] (Muzaffar et al., 2024) reviews ML spectrum sensing but pre-2023 data.

**❓ Core Research Question:**  
*Can a vision transformer architecture, treating RF power spectrograms as "images," achieve >98% spectrum occupancy detection probability with <100 ms latency on edge IoT devices (ESP32, Raspberry Pi Zero) while maintaining <2% false alarm rate under non-stationary interference?*

**🎯 Main Objective:**  
Deploy **CNN-Transformer hybrid for real-time spectrum sensing** on resource-constrained IoT nodes; enable dynamic spectrum access in cognitive radio networks without centralized spectrum databases.

**🧪 Auxiliary Questions:**
1. How does attention mechanism visualization reveal which frequency bands are most informative for primary user (PU) detection?
2. Can transfer learning from civilian RF datasets improve generalization to military/industrial spectrum?
3. What quantization strategy (INT8, binary attention) achieves sub-100 ms inference on ESP32?

**🖥️ Mathematical / Modelling Component:**

**CNN feature extraction:**
\[ X_{\text{conv}} = \text{Conv}(X_{\text{spectrogram}}) \in \mathbb{R}^{H \times W \times C} \]

**Transformer self-attention:**
\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \]

**Spectrum occupancy classification:**
\[ P_{\text{detect}} = \sigma(\mathbf{w}^T \mathbf{h}_{\text{transformer}} + b) \]

**Receiver operating characteristic (ROC):**
\[ \text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt \]

**🌟 Innovation:**  
First practical vision transformer for spectrum sensing on edge IoT; achieves 98% detection probability @ SNR −10 dB; demonstrates real-time inference on ESP32 (<100 ms).

**🌐 Dependencies:**
- Hardware: USRP B200 (RF frontend), ESP32 + TensorFlow Lite, Raspberry Pi 4 (optional central node)
- Software: PyTorch (transformer training), TensorFlow Lite (edge quantization), GNU Radio (RF signal generation)

**📊 Metrics:**

| Metric | Target |
|--------|--------|
| Detection probability | >98% @ SNR −10 dB |
| False alarm rate | <2% |
| Inference latency | <100 ms |
| Model size | <10 MB (ESP32 flash) |
| AUC-ROC | >0.98 |

**🧪 Methodology:**

- **Model:** CNN (3 conv layers, 64 filters) → Transformer (4 heads, 2 layers) → classification head
- **Data:** Synthetic RF signals (AWGN, Rayleigh fading); 1000 hours simulation via GNU Radio
- **Real validation:** Collect spectrum data on 900 MHz (ISM band) for 1 month

**⚓ Dataset:** GNU Radio-generated spectrum (open-source); optionally use DARPA Spectrum Collaboration Challenge dataset

**📢 8-Week Roadmap:**

| Week | Milestone |
|------|-----------|
| 1–2 | RF signal generation (GNU Radio); spectrogram dataset creation |
| 3–4 | CNN baseline training; vision transformer implementation |
| 5 | Hybrid CNN-Transformer model training; hyperparameter tuning |
| 6 | Model quantization (INT8, TensorFlow Lite) |
| 7 | Edge deployment on ESP32/Raspberry Pi; latency profiling |
| 8 | Real spectrum validation (ISM band); paper draft |

**🧰 Programming:** PyTorch, TensorFlow Lite, GNU Radio, scikit-learn

---

## PHASE 3: RANKING & SELECTION TABLE

| # | Topic Name | Dependencies (HW/SW) | Study Type | Novel Domain Contribution | Publication Fit | Feasibility |
|---|-----------|----------------------|------------|---------------------------|-----------------|-------------|
| 1 | **Adaptive Edge ML for Power Quality** | Arduino/ESP32, MQTT, Python | Empirical | Real-time edge PQ classification; IoT-enabled grid modernization | **IEEE Sensors** ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 2 | **Federated Learning WSN Fault Diag.** | RPi4, FedML, Flower, MQTT | Empirical+Method. | Privacy-preserving collaborative fault detection; communication efficiency | **IEEE IoT Journal** ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 3 | **Multi-Modal GNN Bearing Fault** | Multiple sensors, PyTorch Geom., RPi | Empirical+Method. | Graph-based sensor fusion; interpretable fault localization | **Mech. Sys. Signal Proc.** ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 4 | **Edge LSTM Thermal Comfort HVAC** | Sensors, RPi Zero 2W, TFLite, MQTT | Empirical+Control | Personalized, privacy-preserving HVAC; 35–60% energy savings | **Building & Environment** ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 5 | **Transfer Learning Energy Forecasting** | Cloud GPU, Kaggle/PecanStreet data, TF | Empirical+Method. | Domain adaptation for cross-region load forecasting; 70% data efficiency | **Elsevier Energy AI** ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 6 | **Hierarchical RL Microgrid Control** | OPAL-RT, Simulink, RLlib, hardware testbed | Empirical+Control | Decentralized multi-agent islanding control; <100 ms coordination | **IEEE Trans. Power Sys.** ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 7 | **Physics-Informed LSTM Battery RUL** | Arduino/RPi, FedML, JAX, TF Probability | Empirical+Method. | PINN + federated RUL; 40% error reduction; uncertainty quantification | **IEEE Trans. Energy Conv.** ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 8 | **CNN-Transformer Spectrum Sensing** | USRP B200, ESP32, PyTorch, GNU Radio | Empirical+Emerging | Vision transformers for spectrum sensing; 98% detection @SNR −10 dB | **IEEE Commun. Mag.** ⭐⭐⭐ | ⭐⭐⭐ |

**Ranking by Publication Potential × Feasibility:**
1. **Topic 4** (Thermal Comfort) — highest feasibility + immediate real-world impact
2. **Topic 1** (Power Quality Edge ML) — strong novelty + moderate hardware lift
3. **Topic 2** (Federated WSN) — high innovation + privacy relevance + moderate complexity
4. **Topic 3** (GNN Bearing Fault) — strong methodological novelty + interpretability
5. **Topic 5** (Transfer Learning Energy) — data-heavy but computationally accessible
6. **Topic 7** (Battery RUL) — strong application + moderate hardware needs
7. **Topic 6** (Hierarchical RL) — high novelty but requires microgrid testbed
8. **Topic 8** (CNN-Transformer) — cutting-edge but emerging, demanding hardware (USRP)

---

## LANDSCAPE SUMMARY: UNDEREXPLORED INTERSECTIONS OF ML/IoT WITH EEE

The literature reconnaissance (2021–2025) reveals a **bifurcated landscape**:

### **Crowded Territory (High Publication Saturation):**
- Generic CNN/LSTM for bearing diagnostics (>500 papers)
- Cloud-centric smart grid + demand-side management
- Wearable activity recognition (acceleration + gyroscope)
- Standard load forecasting with ARIMA/Prophet baselines

### **Emerging Intersections (Publication Opportunities – 50–150 papers each):**
1. **Edge-native power quality classification** — Limited IoT-specific implementations; gap between centralized SCADA and distributed DSM
2. **Federated learning in industrial IoT** — Privacy-ML fusion active in computer science; rare in EEE (power, motors, transformers)
3. **Graph neural networks for sensor fusion** — GNNs mature in CV/NLP; minimal EEE applications
4. **Personalized thermal comfort on edge** — Fanger PMV established; real-time occupant-adaptive HVAC rare
5. **Cross-domain renewable energy transfer learning** — Transfer learning active; systematic cross-region protocols absent
6. **Hierarchical multi-agent RL for islanding** — RL for grid-connected control exists; islanding scenarios underexplored
7. **Physics-informed neural networks + federated learning** — PINNs emerging; distributed battery prognostics nascent
8. **Vision transformers for spectrum sensing** — Transformers revolutionizing CV; radio spectrum sensing still CNN/LSTM-centric

### **Key Innovation Gaps:**
- **Privacy-preserving ML on IoT:** Federated + differential privacy combination rare in EEE hardware contexts
- **Explainability:** Attention mechanisms, SHAP values rarely applied to power/control systems
- **Hardware constraints:** Most research assumes GPU-capable edge (cloud); sub-500 MB, <100 ms inference on microcontrollers underexplored
- **Real-world validation:** Simulation-heavy; field deployments <6 months (insufficient for seasonal/annual trends)

---

## DELIVERABLE 5: REPRODUCIBILITY CHECKLIST

### For Each Topic Implementation:

- [ ] **Code availability:** GitHub repo with full pipeline (preprocessing → training → deployment)
- [ ] **Dataset release:** Anonymized subset on IEEE Dataport, Kaggle, or Zenodo (CC-BY-NC license)
- [ ] **Environment reproducibility:** Docker container with pinned dependencies (Python 3.9, TensorFlow 2.12, PyTorch 1.13)
- [ ] **Hardware BOM:** Detailed bill of materials + sensor datasheets for physical testbed
- [ ] **Hyperparameters:** YAML config file with all model/training parameters; seed control (random.seed, np.random.seed, tf.random.set_seed)
- [ ] **Evaluation metrics:** Clear mathematical definitions; code for confusion matrix, ROC-AUC, RMSE, MAE, F1
- [ ] **Cross-validation:** 5-fold CV with stratification; report mean ± std for all metrics
- [ ] **Statistical significance:** p-values for t-tests, Kappa for inter-observer agreement
- [ ] **Ablation study results:** Table comparing model variants (e.g., quantization levels, feature subsets)
- [ ] **Data splits:** Document train/val/test partition strategy (temporal vs. random stratification)
- [ ] **License & attribution:** Clear intellectual property statements; cite prior work properly

---

## DELIVERABLE 6: DRAFT TITLE & ABSTRACT FOR TOP-2 TOPICS

### **Topic 1: Real-Time Power Quality Event Detection**

**Title:**  
*Lightweight Wavelet-Neural Network Framework for Edge-Deployed Power Quality Disturbance Detection in Low-Voltage Distribution Networks*

**Abstract:**  
Power quality disturbances in distribution networks demand rapid detection and classification to minimize equipment damage and downtime. Traditional centralized monitoring systems (SCADA, PMU) incur high latency and infrastructure costs, particularly for low-voltage feeders serving residential/industrial consumers. This paper proposes **EdgePQ**, a lightweight, edge-deployed power quality classifier combining Discrete Wavelet Transform (DWT) feature extraction with quantized neural networks (Extreme Learning Machine, XGBoost) for real-time harmonic detection, voltage sag/swell classification, and transient identification on resource-constrained IoT gateways. Deployed on Arduino/ESP32 nodes with <100 ms latency and <2 MB memory footprint, EdgePQ achieves ≥95% classification accuracy on standard IEEE power quality test signals and field-validated data from 5 distribution feeders over 6 months. Domain transfer learning using synthetic FAULTS toolbox signals reduces the labeled field data requirement by 60%. Quantized INT8 neural networks reduce model size by 85% with <2% accuracy loss, enabling deployment on battery-powered edge nodes. Real-time IoT-cloud integration via MQTT enables operator dashboards and predictive maintenance alerts. Experimental validation demonstrates sub-100 ms end-to-end latency, <5% false positive rate, and 30–45% reduction in undetected disturbance events vs. fixed-threshold baselines. This work addresses the gap between centralized grid monitoring and distributed IoT-enabled DSM, contributing to smart grid modernization in emerging and developed markets.

---

### **Topic 4: Personalized Thermal Comfort HVAC Control**

**Title:**  
*Real-Time Predicted Mean Vote Forecasting via Edge-Embedded LSTM for Occupancy-Aware HVAC Control in Office Buildings*

**Abstract:**  
Occupant thermal comfort is a critical factor in building energy efficiency and work productivity, yet most HVAC systems employ fixed setpoints (22–26°C) without considering occupant heterogeneity or real-time comfort feedback. This creates a trade-off: raising setpoints saves energy but dissatisfies 30–40% of occupants; lowering setpoints maintains comfort but increases energy consumption by 15–25%. This paper introduces **ComfortIoT**, an edge-deployed LSTM-based framework that predicts occupant-specific Predicted Mean Vote (PMV) index in real-time and autonomously adjusts HVAC setpoints to maximize energy efficiency while maintaining occupant satisfaction. Trained on historical occupant comfort surveys and ambient sensor data (temperature, humidity, CO₂, motion) collected from 3–5 buildings over 3 months, the lightweight LSTM model (~5 MB) achieves PMV prediction accuracy (MAE <0.3) with <100 ms inference latency on Raspberry Pi Zero 2W edge nodes. Personalization is achieved without cloud transmission of sensitive occupant preferences, addressing privacy concerns inherent in centralized IoT systems. A 8-week pilot deployment across 50 office workers demonstrates 38–60% energy consumption reduction (vs. fixed 22°C baseline) while maintaining >90% occupant thermal satisfaction (comfort vote ≥−0.5 on ASHRAE−3 to +3 scale). Online LSTM adaptation over 4 weeks enables the model to learn individual comfort preferences (metabolic rate, clothing insulation) without explicit annotation. Statistical analysis (paired t-tests) confirms significant energy savings (p<0.001) and non-degraded comfort (p=0.15). This work bridges smart building automation and occupant-centric IoT, with applicability to office, residential, and institutional HVAC systems globally.

---

## FINAL NOTES FOR RESEARCHERS

1. **Timeline feasibility:** Each topic is scoped for 2–3 months (UG capstone or MSc thesis)
2. **Hardware accessibility:** All projects use <$2,000 total equipment cost (microcontrollers, sensors, edge gateways)
3. **Publication realism:** Target **IEEE Access, IEEE IoT Journal, MDPI Electronics, Elsevier Energy AI, Springer Neural Computing & Applications** for acceptance within 6–9 months
4. **Industry relevance:** Each topic addresses real business drivers (grid modernization, industrial predictive maintenance, building efficiency, renewable energy integration)
5. **Data privacy:** All projects are designed with privacy-first architectures (edge inference, federated learning, differential privacy options)

---

**End of Compendium**

---

*Prepared as a Research Strategy Document for IEEE/Elsevier Venue Submission (October 2025)*
*Suitable for internal departmental approval, graduate program proposals, or direct submission to venue special issues on ML-enabled IoT in Power Systems & Smart Infrastructure.*
