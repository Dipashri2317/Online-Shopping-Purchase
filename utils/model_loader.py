import json
from pathlib import Path

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"


@st.cache_data
def load_metadata():

    metadata_path = MODEL_DIR / "metadata.json"

    if not metadata_path.exists():

        raise FileNotFoundError(
            "model/metadata.json is missing. "
            "Run `python train_models.py` first."
        )

    with metadata_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


@st.cache_resource
def load_models():

    model_files = {

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

    loaded = {}

    for model_name, filename in model_files.items():

        model_path = MODEL_DIR / filename

        if model_path.exists():

            loaded[model_name] = joblib.load(
                model_path
            )

    return loaded