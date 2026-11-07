
"""Model loading and evaluation utilities."""

import joblib
import numpy as np
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from config import MODEL_DIR, MODEL_FILES


@st.cache_resource
def load_models():
    """
    Load existing trained models from the model directory.

    The application does not retrain models when it starts.
    """

    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if model_path.exists():
            loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


def evaluate_model(model, X, y):
    """
    Generate predictions and calculate evaluation metrics.
    """

    predictions = model.predict(X)

    scores = None

    # Use predicted probabilities for ROC-AUC when available.
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]

    # Otherwise use the model decision function.
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)

    metrics = {
        "Accuracy": accuracy_score(y, predictions),

        "AUC": (
            roc_auc_score(y, scores)
            if scores is not None
            else np.nan
        ),

        "Precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y,
            predictions
        )
    }

    return predictions, scores, metrics
