# Intelligent Fault Classification & Localization
### A Hybrid CNN-LSTM Framework for Transmission Line Protection

This project focuses on the automated detection, classification, and precise localization of faults in electrical transmission lines. By fusing **1D-Convolutional Neural Networks (CNN)** for spatial feature extraction with **Long Short-Term Memory (LSTM)** networks for temporal sequence analysis, the model achieves high accuracy even under varying noise and loading conditions.

---

## Project Architecture

The workflow is divided into two distinct phases:

1.  **Data Generation (MATLAB/Simulink):** High-fidelity simulation of various fault scenarios at different distances, inception angles, and fault resistances to create a comprehensive dataset.
2.  **Deep Learning Model (Python):** A hybrid architecture built using **TensorFlow/Keras** to classify the fault type (e.g., AG, BC, ABC) and regress the exact fault location (km).

---

## Technical Workflow

### 1. The Data Foundation (Completed)
The dataset is established using a power system model in **MATLAB/Simulink**. This phase focuses on generating three-phase voltage ($V_{abc}$) and current ($I_{abc}$) signals.

**The simulation accounts for:**
* **Fault Types:** 11 distinct states (10 short-circuit types + 1 Normal state).
* **Inception Angle:** Range of $0^\circ$ to $330^\circ$ to capture transient variety.
* **Fault Resistance ($R_f$):** Multiple values to simulate high and low impedance faults $[0.001\Omega, 1\Omega, 10\Omega, 50\Omega, 100\Omega]$.
* **Fault Location:** Systematic intervals (1 km to 300 km) along the line for precise localization data.

### 2. The Hybrid Framework (Planned)
The proposed architecture follows a sequential feature-processing pipeline:

* **Spatial Feature Extraction (CNN):** 1D-Convolutional layers analyze the instantaneous signatures of the three-phase signals to identify patterns unique to specific fault categories.
* **Temporal Dependency Mapping (LSTM):** Extracted features are fed into LSTM units to learn the time-series progression, essential for distinguishing between permanent faults and temporary line transients.

---

## Project Status: Work in Progress

- [x] Transmission Line Modeling in Simulink
- [x] Automated Data Generation Script
- [ ] Data Conversion (.mat to .npy)
- [ ] Data Preprocessing and Normalization
- [ ] CNN-LSTM Model Architecture Design
- [ ] Hyperparameter Tuning and Optimization
- [ ] Model Training and Evaluation
- [ ] Performance Analysis and Visualization

## Tech Stack
* **Simulation:** MATLAB, Simulink, Simscape Electrical
* **Deep Learning:** Python, TensorFlow, Keras, NumPy
* **Data Handling:** Pandas, SciPy (for `.mat` parsing)
* **Visualization:** Matplotlib, Seaborn