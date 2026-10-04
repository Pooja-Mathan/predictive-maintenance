"""Data loading, cleaning, and feature engineering for the AI4I 2020 dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "machine_failure"

# Original column name -> clean snake_case name
RENAME_MAP = {
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": TARGET,
}

# Columns that must NOT be used as features:
#  - identifiers carry no signal
#  - individual failure modes leak the target (target = OR of these)
DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

CATEGORICAL = ["type"]
NUMERIC = [
    "air_temp", "process_temp", "rpm", "torque", "tool_wear",
    "temp_diff", "power_w", "wear_torque",
]
FEATURES = CATEGORICAL + NUMERIC


def load_data(path: str) -> pd.DataFrame:
    """Read the raw CSV."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. See data/README.md for download instructions."
        ) from e


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop useless/leaky columns, rename the rest, remove duplicates and NaNs."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.rename(columns=RENAME_MAP)
    return df.drop_duplicates().dropna().reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-inspired features tied to the dataset's failure modes."""
    df = df.copy()
    df["temp_diff"] = df["process_temp"] - df["air_temp"]          # heat dissipation (HDF)
    df["power_w"] = df["torque"] * df["rpm"] * 2 * np.pi / 60.0    # mechanical power (PWF)
    df["wear_torque"] = df["tool_wear"] * df["torque"]             # overstrain (OSF)
    return df


def prepare_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load -> clean -> engineer features. Returns (X, y)."""
    df = add_features(clean_data(load_data(path)))
    return df[FEATURES], df[TARGET].astype(int)


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode the categorical column, scale numerics."""
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
