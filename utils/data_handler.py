from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "online_shoppers_intention.csv"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"


def convert_target_values(series):

    if series.isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    if pd.api.types.is_bool_dtype(series):

        converted = series.astype(int)

    elif pd.api.types.is_numeric_dtype(series):

        converted = pd.to_numeric(
            series,
            errors="coerce"
        )

    else:

        mapping = {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
            "purchase": 1,
            "no purchase": 0
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

        raise ValueError(
            "Target values must contain only 0 and 1."
        )

    return converted


def load_default_dataset():

    if not TEST_DATA_PATH.exists():

        raise FileNotFoundError(
            "test_data.csv was not found. "
            "Run `python train_models.py` first."
        )

    return pd.read_csv(TEST_DATA_PATH)


def validate_dataset(
    data,
    feature_columns,
    target_column
):

    if data.empty:

        return False, "The uploaded dataset is empty.", data

    required = feature_columns + [target_column]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        return (
            False,
            f"Missing required columns: {missing}",
            data
        )

    data = data.copy()

    try:

        data[target_column] = convert_target_values(
            data[target_column]
        )

    except ValueError as error:

        return False, str(error), data

    categorical_columns = [
        "Month",
        "VisitorType",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "Weekend"
    ]

    numeric_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if data[feature_columns + [target_column]].isna().any().any():

        return (
            False,
            "Dataset contains missing or non-numeric feature values.",
            data
        )

    if data[target_column].nunique() != 2:

        return (
            False,
            "Dataset must contain both target classes (0 and 1).",
            data
        )

    return True, "Dataset validation successful.", data