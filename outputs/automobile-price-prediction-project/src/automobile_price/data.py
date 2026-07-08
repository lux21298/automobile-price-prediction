from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_PATH, TARGET_COLUMN


FEATURE_NAMES = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]

NUMERIC_COLUMNS = [
    "symboling",
    "normalized-losses",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-size",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
]

CATEGORICAL_COLUMNS = [
    "make",
    "fuel-type",
    "aspiration",
    "num-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "engine-type",
    "num-cylinders",
    "fuel-system",
]


def load_raw_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw UCI Automobile data and attach column names."""
    return pd.read_csv(
        path,
        sep=",",
        names=FEATURE_NAMES,
        header=None,
        na_values="?",
        encoding="latin-1",
    )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns and remove rows without a target price."""
    cleaned = df.copy()

    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    return cleaned


def load_modeling_data(path=DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return features and target for model training."""
    data = prepare_data(load_raw_data(path))
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    return X, y

