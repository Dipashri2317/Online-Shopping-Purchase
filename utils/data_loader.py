
"""Dataset loading, target conversion and validation utilities."""

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split

from config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


def convert_target_values(series):
    """
    Convert supported target values into binary values:
    0 = No Purchase
    1 = Purchase
    """

    if series.isna().any():
        raise ValueError("Target column contains missing values.")

    if pd.api.types.is_bool_dtype(series):
        converted = series.astype(int)

    elif pd.api.types.is_numeric_dtype(series):
        converted = pd.to_numeric(series, errors="coerce")

    else:
        mapping = {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0
        }

        converted = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

    if converted.isna().any():
        raise ValueError(
            "Target values must be True/False, yes/no, or 0/1."
        )

    converted = converted.astype(int)

    if not set(converted.unique()).issubset({0, 1}):
        raise ValueError("Target values must contain only 0 and 1.")

    return converted


@st.cache_data
def load_default_dataset():
    """
    Load the original dataset and return the same
    stratified 20% test split used for model evaluation.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates().copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset must contain '{TARGET_COLUMN}'."
        )

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Dataset is missing features: {missing_features}"
        )

    df[TARGET_COLUMN] = convert_target_values(
        df[TARGET_COLUMN]
    )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Same stratified test split used when the models were trained.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test

    return test_data.reset_index(drop=True)


@st.cache_data
def create_sample_dataset():
    """Create the downloadable sample test dataset."""

    return load_default_dataset()


def validate_dataset(df):
    """Validate the input dataset before model evaluation."""

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"

    if df.empty:
        return False, "The uploaded dataset is empty."

    if df[required_columns].isna().any().any():
        return False, "The dataset contains missing values."

    if df[TARGET_COLUMN].nunique() != 2:
        return (
            False,
            "The dataset must contain both target classes (0 and 1)."
        )

    return True, "Dataset validated successfully."
