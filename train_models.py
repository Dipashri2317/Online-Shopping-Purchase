from pathlib import Path
import json
import warnings

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from utils.data_handler import convert_target_values


warnings.filterwarnings("ignore", category=FutureWarning)


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "online_shoppers_intention.csv"

MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)


FEATURE_COLUMNS = [

    "Administrative",
    "Administrative_Duration",

    "Informational",
    "Informational_Duration",

    "ProductRelated",
    "ProductRelated_Duration",

    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",

    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",

    "Month",
    "VisitorType",
    "Weekend"
]


TARGET_COLUMN = "Revenue"


CATEGORICAL_COLUMNS = [

    "Month",
    "VisitorType",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "Weekend"
]


NUMERIC_COLUMNS = [

    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_COLUMNS
]


MODEL_FILES = {

    "Logistic Regression":
        "logistic_regression.pkl",

    "Decision Tree":
        "decision_tree.pkl",

    "KNN":
        "knn.pkl",

    "Gaussian Naive Bayes":
        "gaussian_naive_bayes.pkl",

    "SVM":
        "svm.pkl"
}


def load_dataset():

    if not DATASET_PATH.exists():

        print(
            "Dataset CSV not found. "
            "Trying to download it from UCI..."
        )

        from download_dataset import main as download_dataset

        download_dataset()

    data = pd.read_csv(DATASET_PATH)

    missing = [

        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    data = data[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ].drop_duplicates().copy()

    data[TARGET_COLUMN] = convert_target_values(
        data[TARGET_COLUMN]
    )

    for column in NUMERIC_COLUMNS:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if data[FEATURE_COLUMNS].isna().any().any():

        raise ValueError(
            "Dataset contains missing or invalid feature values."
        )

    return data


def build_preprocessor():

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_transformer,
                NUMERIC_COLUMNS
            ),

            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_COLUMNS
            )

        ],

        remainder="drop",

        verbose_feature_names_out=False
    )

    return preprocessor


def main():

    data = load_dataset()

    X = data[FEATURE_COLUMNS]

    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    # Save a fixed test dataset for the Streamlit application.

    test_data = X_test.copy()

    test_data[TARGET_COLUMN] = y_test

    test_data.to_csv(
        BASE_DIR / "test_data.csv",
        index=False
    )

    estimators = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            ),

        "KNN":
            KNeighborsClassifier(
                n_neighbors=5
            ),

        "Gaussian Naive Bayes":
            GaussianNB(),

        "SVM":
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=42,
                kernel="rbf"
            )
    }

    for model_name, estimator in estimators.items():

        print(f"Training {model_name}...")

        pipeline = Pipeline(

            steps=[

                (
                    "preprocessor",
                    build_preprocessor()
                ),

                (
                    "classifier",
                    estimator
                )

            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        output_file = (
            MODEL_DIR / MODEL_FILES[model_name]
        )

        joblib.dump(
            pipeline,
            output_file
        )

        print(
            f"Saved: {output_file.name}"
        )

    metadata = {

        "dataset":
            "UCI Online Shoppers Purchasing Intention Dataset",

        "target_column":
            TARGET_COLUMN,

        "target_mapping": {
            "0": "No Purchase",
            "1": "Purchase"
        },

        "features":
            FEATURE_COLUMNS,

        "categorical_features":
            CATEGORICAL_COLUMNS,

        "numeric_features":
            NUMERIC_COLUMNS,

        "models":
            list(MODEL_FILES.keys()),

        "test_size":
            0.20,

        "random_state":
            42,

        "rows_after_duplicate_removal":
            int(len(data)),

        "training_rows":
            int(len(X_train)),

        "testing_rows":
            int(len(X_test))
    }

    with (
        MODEL_DIR / "metadata.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nTraining completed successfully.")

    print(
        f"Rows after duplicate removal: {len(data):,}"
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Testing rows: {len(X_test):,}"
    )

    print(
        "Saved model files and test_data.csv."
    )


if __name__ == "__main__":

    main()