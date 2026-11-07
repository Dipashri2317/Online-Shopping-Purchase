
"""Central configuration for Online Shoppers Purchase Prediction."""

from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset and model directories
DATASET_PATH = BASE_DIR / "online_shoppers_intention.csv"
MODEL_DIR = BASE_DIR / "model"

# Input features used by all five trained models
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

# Target variable
TARGET_COLUMN = "Revenue"

# Saved trained model filenames
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "SVM": "svm.pkl"
}

# Evaluation metrics
METRIC_NAMES = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]
