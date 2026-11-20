# MaintenAI – ML-Based Industrial Predictive Maintenance and Failure Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/UCI-AI4I%202020-green.svg)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

**MaintenAI** is an end-to-end, enterprise-grade industrial predictive maintenance and failure detection system. Powered by physics-informed feature engineering and machine learning, MaintenAI identifies impending machine failures, classifies specific component failure modes (multilabel), estimates failure probabilities across operating fleets, monitors real-time telemetry, and prescribes targeted maintenance actions.

---

## 📑 Table of Contents
1. [System Architecture](#system-architecture)
2. [Dataset Specifications](#dataset-specifications)
3. [Physics-Informed Feature Engineering](#physics-informed-feature-engineering)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
   - [A. Binary Failure Prediction](#a-binary-failure-prediction)
   - [B. Multilabel Failure Mode Classification](#b-multilabel-failure-mode-classification)
   - [C. Class Imbalance & SMOTE Benchmarking](#c-class-imbalance--smote-benchmarking)
5. [The Eight Application Modules](#the-eight-application-modules)
6. [Project Structure](#project-structure)
7. [Installation & Execution Guide](#installation--execution-guide)
8. [Important Industrial Constraints & Disclaimer](#important-industrial-constraints--disclaimer)

---

## 🏛️ System Architecture

MaintenAI bridges operational technology (OT) telemetry with machine learning pipelines:

```
[ Sensor Telemetry (Air Temp, Process Temp, Speed, Torque, Tool Wear, Type) ]
                                    │
                                    ▼
       [ Physics-Informed Feature Engineering (PWF, HDF, OSF indicators) ]
                                    │
                                    ▼
       [ Scikit-Learn Preprocessing Pipeline (StandardScaler + OneHotEncoder) ]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
    [ Binary Failure Classifiers ]       [ Multilabel Classifier ]
     - XGBoost (Top Model)                - MultiOutput(Random Forest)
     - Random Forest                      - TWF, HDF, PWF, OSF, RNF
     - Decision Tree
     - Logistic Regression
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
           [ Failure Probability & Risk Tier Segmentation ]
             (Low Risk < 25%, Moderate 25-65%, High > 65%)
                                    │
                                    ▼
         [ Rule-Based Maintenance Recommendation System ]
                                    │
                                    ▼
              [ SHAP Local & Global Explainability ]
                                    │
                                    ▼
                 [ Streamlit Enterprise Web Interface ]
```

---

## 📊 Dataset Specifications

The system utilizes the authentic **AI4I 2020 Predictive Maintenance Dataset** from the official UCI Machine Learning Repository:
- **Total records:** 10,000 synthetic industrial milling machines
- **Features:** 14 total columns (excluding unique identification columns `UID` and `Product ID` during model inference to prevent target leakage).
- **Target Variable:** `Machine failure` (Binary: 0 for Healthy, 1 for Failure; 339 failure events, 3.39% minority class).
- **Component Failure Modes (Multilabel):**
  - **TWF (Tool Wear Failure):** 46 occurrences
  - **HDF (Heat Dissipation Failure):** 115 occurrences
  - **PWF (Power Failure):** 95 occurrences
  - **OSF (Overstrain Failure):** 98 occurrences
  - **RNF (Random Failure):** 19 occurrences

---

## ⚙️ Physics-Informed Feature Engineering

Based on the underlying physical equations governing milling machine mechanics:

1. **Temperature Difference ($\Delta T$):**
   $$\Delta T = T_{\text{process}} - T_{\text{air}}$$
   *Physical Meaning:* Heat dissipation failure (HDF) occurs when $\Delta T < 8.6\text{ K}$ and rotational speed $< 1380\text{ rpm}$.
2. **Mechanical Power Delivered ($P_{\text{kW}}$):**
   $$P = \frac{\tau \cdot \omega}{1000} = \frac{\tau \cdot (N \cdot 2\pi / 60)}{1000}$$
   *Physical Meaning:* Power failure (PWF) occurs when delivered power drops below $3.5\text{ kW}$ or surges above $9.0\text{ kW}$.
3. **Cumulative Tool Wear Strain:**
   $$\text{Strain} = \tau \times \text{Tool wear}$$
   *Physical Meaning:* Overstrain failure (OSF) triggers when this product exceeds structural mechanical limits ($11,000$ for Type L, $12,000$ for Type M, $13,000\text{ min}\cdot\text{Nm}$ for Type H).
4. **Speed-to-Torque Ratio:**
   Captures the active gearing and mechanical operating envelope.

---

## 🧠 Machine Learning Pipeline

### A. Binary Failure Prediction
Four distinct model architectures were trained and evaluated on an 80/20 stratified split:

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **98.85%** | **87.69%** | **83.82%** | **0.8571** | **0.9826** | **0.9023** |
| **Random Forest** | 98.40% | 69.41% | 86.76% | 0.7712 | 0.9828 | 0.9095 |
| **Decision Tree** | 96.65% | 42.07% | 89.71% | 0.5728 | 0.9241 | 0.8691 |
| **Logistic Regression** | 95.80% | 35.36% | 94.12% | 0.5141 | 0.9842 | 0.9191 |

*Note on Model Selection:* XGBoost was selected as the default production model because it maintains the optimal balance between high precision (87.69%) and high recall (83.82%), resulting in the highest F1-score (0.8571) and PR-AUC.

### B. Multilabel Failure Mode Classification
Component failures are **not mutually exclusive**; 24 historical machines suffered multiple simultaneous failure modes. A `MultiOutputClassifier(RandomForestClassifier)` models all five failure types independently:
- **Hamming Loss:** $0.0108$
- **HDF F1-Score:** $1.0000$
- **PWF F1-Score:** $1.0000$
- **OSF F1-Score:** $1.0000$

### C. Class Imbalance & SMOTE Benchmarking
Machine failures account for only 3.39% of the dataset. We benchmarked cost-sensitive class weighting vs. SMOTE (applied strictly on training data):
- Cost-sensitive weighting allowed tree ensembles to maintain precision above 87% without generating excessive false positive maintenance alerts.
- SMOTE increased sensitivity for linear models but introduced false positives for tree ensembles on boundary points. It identifies the nearest minority-class neighbours and creates new samples through interpolation between them. This improves minority-class representation and helps the model learn failure patterns more effectively. SMOTE is applied only to the training data to prevent data leakage.

---

## 🖥️ The Eight Application Modules

1. **Overview Dashboard (`pages/overview.py`):**
   Fleet-wide KPIs, healthy vs. failure percentages, machine variant distribution, sensor baseline statistics, and dataset integrity metrics.
2. **Machine Failure Prediction (`pages/failure_prediction.py`):**
   Interactive inference console with input sliders, real-time physics validation, preset industrial test scenarios, failure probability gauge, and model switcher.
3. **Failure Type Analysis (`pages/failure_analysis.py`):**
   Historical multilabel co-occurrence matrix, sensor boxplots per failure mode, and live component failure mode diagnostics for any input.
4. **Failure Probability Monitoring (`pages/risk_monitoring.py`):**
   Dynamic risk surveillance table for all 10,000 machines, threshold calibration sliders (Low, Moderate, High risk), and side-by-side machine comparison workbench.
5. **Sensor Data Visualization (`pages/sensor_analytics.py`):**
   Interactive Plotly distributions, correlation heatmap, 2D bivariate scatter with marginals, 3D operational envelope, and Tukey IQR outlier detection.
6. **ML Model Comparison (`pages/model_comparison.py`):**
   Comparative metric tables, grouped bar charts, radar charts, ROC curves, Precision-Recall curves, confusion matrices, and SMOTE benchmark.
7. **Maintenance Recommendation System (`pages/maintenance_recommendations.py`):**
   Hybrid physics-rule + ML engine prescribing targeted maintenance actions, technician dispatch priorities (Critical, High, Medium, Low), and fleet work order queues.
8. **Model Insights and Explainability (`pages/model_explainability.py`):**
   Global feature importances (RF MDI vs. XGBoost Gain) and local SHAP instance waterfall charts showing exact sensor attributions for any machine.

---

## 📁 Project Structure

```
MaintenAI/
├── app.py                           # Main Streamlit application entry point
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation and architecture guide
├── data/
│   └── ai4i2020.csv                 # Authentic AI4I 2020 dataset
├── models/
│   ├── xgboost.pkl                  # Trained XGBoost classifier
│   ├── random_forest.pkl            # Trained Random Forest classifier
│   ├── decision_tree.pkl            # Trained Decision Tree classifier
│   ├── logistic_regression.pkl      # Trained Logistic Regression classifier
│   ├── multilabel_classifier.pkl    # Trained Multilabel Failure Mode classifier
│   ├── preprocessing_pipeline.pkl   # Fitted StandardScaler + OneHotEncoder
│   ├── feature_names.pkl            # Transformed feature name registry
│   └── evaluation_metrics.pkl       # Serialized benchmark test metrics
├── src/
│   ├── data_preprocessing.py        # Loading, splitting, and scaling pipelines
│   ├── feature_engineering.py       # Physics-informed derived features
│   ├── train_models.py              # Model training and artifact export script
│   ├── failure_prediction.py        # Binary failure inference engine
│   ├── failure_type_prediction.py   # Multilabel component failure inference
│   ├── risk_assessment.py           # Fleet risk scoring and tier categorization
│   ├── maintenance_recommendations.py# Rule-based work order recommendation engine
│   ├── model_evaluation.py          # Metric calculation, curves, and reports
│   └── explainability.py            # Global importance & local SHAP attributions
├── pages/
│   ├── overview.py                  # Module 1: Overview Dashboard
│   ├── failure_prediction.py        # Module 2: Machine Failure Prediction
│   ├── failure_analysis.py          # Module 3: Failure Type Analysis
│   ├── risk_monitoring.py           # Module 4: Failure Probability Monitoring
│   ├── sensor_analytics.py          # Module 5: Sensor Data Visualization
│   ├── model_comparison.py          # Module 6: ML Model Comparison
│   ├── maintenance_recommendations.py# Module 7: Maintenance Recommendations
│   └── model_explainability.py      # Module 8: Model Explainability
└── utils/
    ├── visualizations.py            # Plotly chart generators & themes
    └── helpers.py                   # Caching, styling, presets, and validation
```
