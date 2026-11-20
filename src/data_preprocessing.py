"""
Data Preprocessing Pipeline Module for MaintenAI.
Handles data loading, validation, leakage prevention, stratified splitting,
and sklearn-compatible transformation pipelines.
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from src.feature_engineering import (
    add_engineered_features,
    RAW_FEATURE_COLS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    FAILURE_TYPE_COLS,
)


def load_dataset(csv_path: str = "data/ai4i2020.csv") -> pd.DataFrame:
    """Loads the authentic AI4I 2020 dataset from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run data download first.")
    df = pd.read_csv(csv_path)
    return df


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates an automated data quality summary."""
    return {
        "total_records": len(df),
        "total_features": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "failure_count": int(df[TARGET_COL].sum()) if TARGET_COL in df.columns else 0,
        "failure_rate_pct": float(df[TARGET_COL].mean() * 100) if TARGET_COL in df.columns else 0.0,
    }


def prepare_features_and_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extracts raw predictive features and targets while strictly avoiding target leakage.
    - Excludes 'UID', 'Product ID'
    - Excludes 'Machine failure' and failure types ('TWF', 'HDF', 'PWF', 'OSF', 'RNF') from X
    """
    df_engineered = add_engineered_features(df)

    # Features: only sensor features and engineered features, NO target or ID columns
    feature_cols = [c for c in NUMERICAL_COLS + CATEGORICAL_COLS if c in df_engineered.columns]
    X = df_engineered[feature_cols].copy()

    # Binary Target
    y_binary = df[TARGET_COL].copy()

    # Multilabel Target
    y_multilabel = df[FAILURE_TYPE_COLS].copy()

    return X, y_binary, y_multilabel


def create_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for numerical scaling and categorical encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            (
                "cat",
                OneHotEncoder(categories=[["L", "M", "H"]], handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_stratified_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train-test split to preserve class imbalance ratio in both splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def apply_smote_resampling(
    X_train_transformed: np.ndarray, y_train: pd.Series, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) strictly to the training data.
    Never applied to validation or test data to avoid optimistic data leakage.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)
    return X_resampled, y_resampled
