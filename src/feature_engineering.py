"""
Feature Engineering Module for MaintenAI.
Extracts physics-based domain features from operational sensor measurements.
Domain calculations follow standard mechanical engineering and AI4I dataset physics:
- Temperature Difference [K]: Process Temp - Air Temp (Heat dissipation dynamics)
- Power [kW]: Torque * Rotational Speed * 2 * pi / 60 / 1000 (Mechanical power delivered)
- Tool Wear Strain [Nm*min]: Torque * Tool wear (Cumulative cutting stress)
- Speed-to-Torque Ratio: Rotational speed / Torque (Operating mechanical regime)
- Physics-based threshold risk indicators
"""

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain and physics-informed features from raw machine sensor measurements.
    Works for both single-row inputs and batch DataFrames.
    """
    data = df.copy()

    # Temperature difference (Process - Air)
    data["Temp_Difference"] = data["Process temperature"] - data["Air temperature"]

    # Mechanical Power delivered in Kilowatts
    # P = (Torque in Nm * Rotational speed in rad/s) / 1000
    # rad/s = rpm * 2 * pi / 60
    data["Power_kW"] = (data["Torque"] * data["Rotational speed"] * 2 * np.pi) / (60 * 1000)

    # Tool Wear Strain (Cumulative mechanical burden on cutting tool)
    data["Tool_Wear_Strain"] = data["Torque"] * data["Tool wear"]

    # Speed to Torque Ratio
    data["Speed_Torque_Ratio"] = data["Rotational speed"] / (data["Torque"].clip(lower=0.1))

    # Physics-based indicators (per UCI AI4I 2020 domain specification)
    # HDF risk: Temp difference < 8.6 K and Rotational speed < 1380 rpm
    data["HDF_Risk_Flag"] = (
        (data["Temp_Difference"] < 8.6) & (data["Rotational speed"] < 1380)
    ).astype(int)

    # PWF risk: Power < 3.5 kW or Power > 9.0 kW
    data["PWF_Risk_Flag"] = (
        (data["Power_kW"] < 3.5) | (data["Power_kW"] > 9.0)
    ).astype(int)

    # OSF risk: Product of tool wear and torque exceeds capacity threshold
    # Thresholds: L: 11,000, M: 12,000, H: 13,000 min*Nm
    type_col = data["Type"] if "Type" in data.columns else "L"
    threshold = 11000
    if isinstance(type_col, pd.Series):
        threshold = type_col.map({"L": 11000, "M": 12000, "H": 13000}).fillna(11000)
    elif isinstance(type_col, str):
        threshold = {"L": 11000, "M": 12000, "H": 13000}.get(type_col, 11000)

    data["OSF_Risk_Flag"] = (data["Tool_Wear_Strain"] > threshold).astype(int)

    # TWF risk: Tool wear between 200 and 240+ minutes
    data["TWF_Risk_Flag"] = (data["Tool wear"] >= 200).astype(int)

    return data


# Canonical feature definitions
RAW_FEATURE_COLS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

NUMERICAL_COLS = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Temp_Difference",
    "Power_kW",
    "Tool_Wear_Strain",
    "Speed_Torque_Ratio",
    "HDF_Risk_Flag",
    "PWF_Risk_Flag",
    "OSF_Risk_Flag",
    "TWF_Risk_Flag",
]

CATEGORICAL_COLS = ["Type"]

ALL_ENGINEERED_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

FAILURE_TYPE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET_COL = "Machine failure"
