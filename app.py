# ============================================================
# ONLINE SHOPPERS PURCHASE PREDICTION - STREAMLIT APP
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay, classification_report, roc_curve
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Online Shoppers Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "online_shoppers_intention.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

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

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "SVM": "svm.pkl"
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]

# ============================================================
# 3. CUSTOM CSS - MIDNIGHT TEAL + EMERALD THEME
# ============================================================

def apply_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg,#F0F8F5 0%,#F7FBF9 50%,#E8F4EF 100%) !important;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg,#102C35 0%,#126B68 55%,#20A486 100%) !important;
            padding: 28px !important;
            border-radius: 18px !important;
            color: #ffffff !important;
            margin-bottom: 25px !important;
            box-shadow: 0 5px 15px rgba(16,47,96,0.15) !important;
        }
        .hero h1 {
            color: #ffffff !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            margin-bottom: 10px !important;
        }
        .hero p {
            color: #D7FFF2 !important;
            font-size: 1rem !important;
            margin-bottom: 5px !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg,#102C35 0%,#155E59 55%,#168574 100%) !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] span {
            color: #ffffff !important;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg,#ffffff 0%,#E4F3EF 100%) !important;
            border: 1px solid #CFE5DC !important;
            padding: 10px !important;
            border-radius: 12px !important;
            min-height: 115px;
            box-shadow: 0 3px 10px rgba(16,47,96,0.06);
        }
        div[data-testid="stMetricLabel"] {
            color: #37645C !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #102C35 !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.08) !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            background: rgba(255,255,255,0.12) !important;
            border: 1px dashed #CFE5DC !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section span,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section small,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section p {
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            background: #ffffff !important;
            color: #153D3A !important;
            border: 1px solid #CFE5DC !important;
            border-radius: 8px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] button span,
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] button p {
            color: #153D3A !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            background: #ffffff !important;
            border-radius: 8px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
            color: #153D3A !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"] {
            background: #ffffff !important;
            color: #153D3A !important;
        }
        .stButton button, .stDownloadButton button {
            background: linear-gradient(135deg,#126B68,#168574) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            background: linear-gradient(135deg,#102C35,#126B68) !important;
            color: #ffffff !important;
            border: none !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #CFE5DC !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stTable"] {
            color: #153D3A !important;
        }
        div[data-baseweb="select"] > div {
            border-radius: 8px !important;
            border-color: #BFDCD1 !important;
        }
        .main h1, .main h2, .main h3 {
            color: #153D3A;
        }
        .hero h1, .hero p {
            color: #ffffff !important;
        }
        .footer {
            background: linear-gradient(135deg,#102C35 0%,#126B68 100%) !important;
            color: #ffffff !important;
            text-align: center !important;
            padding: 18px !important;
            border-radius: 12px !important;
            margin-top: 35px !important;
        }
        .footer p {
            color: #ffffff !important;
            margin: 0 !important;
            font-size: 0.9rem !important;
        }
        @media (max-width:768px) {
            .hero { padding: 20px !important; }
            .hero h1 { font-size: 1.7rem !important; }
            div[data-testid="stMetric"] { height: auto; }
        }
        .stApp .hero {
            background: linear-gradient(135deg,#102C35 0%,#126B68 55%,#20A486 100%) !important;
            padding: 28px !important;
            border-radius: 18px !important;
            margin-bottom: 25px !important;
        }
        .stApp .hero h1 { color: #ffffff !important; }
        .stApp .hero p { color: #D7FFF2 !important; }
        .stApp .footer {
            background: linear-gradient(135deg,#102C35 0%,#126B68 100%) !important;
            color: #ffffff !important;
            text-align: center !important;
            padding: 18px !important;
            border-radius: 12px !important;
            margin-top: 35px !important;
        }
        .stApp .footer p { color: #ffffff !important; }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():
    st.html("""
        <div class="hero">
            <h1>🛒 Online Shoppers Purchase Prediction</h1>
            <p>Developed using Python, Scikit-learn and Streamlit</p>
            <p>Dataset: UCI Online Shoppers Purchasing Intention Dataset</p>
            <p>Machine Learning Assignment 2</p>
        </div>
    """)


def render_footer():
    st.html("""
        <div class="footer">
            <p>Online Shoppers Purchase Prediction | Machine Learning Assignment 2</p>
        </div>
    """)

# ============================================================
# 4. TARGET CONVERSION
# ============================================================

def convert_target_values(series):
    if series.isna().any():
        raise ValueError("Target column contains missing values.")

    if pd.api.types.is_bool_dtype(series):
        converted = series.astype(int)
    elif pd.api.types.is_numeric_dtype(series):
        converted = pd.to_numeric(series, errors="coerce")
    else:
        mapping = {
            "true": 1, "false": 0, "1": 1, "0": 0,
            "yes": 1, "no": 0
        }
        converted = series.astype(str).str.strip().str.lower().map(mapping)

    if converted.isna().any():
        raise ValueError("Target values must be True/False, yes/no, or 0/1.")

    converted = converted.astype(int)
    if not set(converted.unique()).issubset({0, 1}):
        raise ValueError("Target values must contain only 0 and 1.")

    return converted


# ============================================================
# 5. DEFAULT TEST DATASET AND SAMPLE DOWNLOAD
# ============================================================

@st.cache_data
def load_default_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates().copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain '{TARGET_COLUMN}'.")

    df[TARGET_COLUMN] = convert_target_values(df[TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Same stratified test split used when the models were trained.
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test
    return test_data.reset_index(drop=True)


@st.cache_data
def create_sample_dataset():
    return load_default_dataset()


# ============================================================
# 6. LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_models():
    loaded = {}

    for model_name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            loaded[model_name] = joblib.load(path)

    return loaded


# ============================================================
# 7. VALIDATION AND EVALUATION FUNCTIONS
# ============================================================

def validate_dataset(df):
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]

    if missing:
        return False, f"Missing required columns: {missing}"

    if df.empty:
        return False, "The uploaded dataset is empty."

    if df[required].isna().any().any():
        return False, "The dataset contains missing values."

    if df[TARGET_COLUMN].nunique() != 2:
        return False, "The dataset must contain both target classes (0 and 1)."

    return True, "Dataset validated successfully."


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    scores = None

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, scores) if scores is not None else np.nan,
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1 Score": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions)
    }

    return predictions, scores, metrics


# ============================================================
# 8. START APP AND LOAD MODELS
# ============================================================

apply_custom_styles()
render_header()

models = load_models()

if not models:
    st.error(
        "No trained models found. Check that all five .pkl files "
        "are inside the model folder."
    )
    st.stop()


# ============================================================
# 9. SIDEBAR
# ============================================================

st.sidebar.title("Online Shopping Prediction")
st.sidebar.markdown(
    "Upload a CSV file to evaluate the trained machine learning models."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

st.sidebar.markdown("""
**CSV Requirements**
- File format: CSV
- Must contain all 17 input features
- Target column can have any name
- Target values must be 0 or 1
- Must contain both target classes
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Sample Dataset")

try:
    sample_dataset = create_sample_dataset()
    st.sidebar.download_button(
        label="Download Sample Dataset",
        data=sample_dataset.to_csv(index=False).encode("utf-8"),
        file_name="online_shoppers_test_data.csv",
        mime="text/csv",
        width="stretch"
    )
except Exception as error:
    st.sidebar.error(f"Sample dataset unavailable: {error}")


# ============================================================
# 10. LOAD DEFAULT OR UPLOADED DATASET
# ============================================================

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        dataset_source = "Uploaded Dataset"
    except Exception as error:
        st.error(f"Unable to read uploaded CSV file: {error}")
        st.stop()

    target_options = [
        col for col in data.columns
        if col not in FEATURE_COLUMNS
    ]

    if not target_options:
        st.error("No target column found. Include the actual target in your CSV.")
        st.stop()

    default_index = (
        target_options.index(TARGET_COLUMN)
        if TARGET_COLUMN in target_options else 0
    )

    selected_target = st.sidebar.selectbox(
        "Select Target Column",
        options=target_options,
        index=default_index
    )

    if selected_target != TARGET_COLUMN:
        if TARGET_COLUMN in data.columns:
            st.error(
                f"The CSV already contains a '{TARGET_COLUMN}' column. "
                "Remove the extra target column or select Revenue."
            )
            st.stop()
        data = data.rename(columns={selected_target: TARGET_COLUMN})
else:
    try:
        data = load_default_dataset()
        dataset_source = "Default Test Dataset"
    except Exception as error:
        st.error(f"Unable to load default dataset: {error}")
        st.stop()


# ============================================================
# 11. VALIDATE DATASET
# ============================================================

if TARGET_COLUMN not in data.columns:
    st.error(f"Target column '{TARGET_COLUMN}' was not found.")
    st.stop()

try:
    data[TARGET_COLUMN] = convert_target_values(data[TARGET_COLUMN])
except Exception as error:
    st.error(str(error))
    st.stop()

is_valid, validation_message = validate_dataset(data)
if not is_valid:
    st.error(validation_message)
    st.stop()

X_test = data[FEATURE_COLUMNS].copy()
y_test = data[TARGET_COLUMN].copy()

st.success(f"{dataset_source} loaded successfully.")


# ============================================================
# 12. DATASET OVERVIEW AND CLASS DISTRIBUTION
# ============================================================

st.markdown("## Dataset Overview")

left_column, right_column = st.columns([1, 1])

total_records = len(data)
total_features = len(FEATURE_COLUMNS)
no_purchase_count = int((y_test == 0).sum())
purchase_count = int((y_test == 1).sum())

with left_column:
    st.metric("Total Records", total_records)
    st.metric("Input Features", total_features)

    st.markdown("### Dataset Preview")
    st.dataframe(data.head(), width="stretch")

with right_column:
    st.markdown("### Class Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["No Purchase", "Purchase"],
        [no_purchase_count, purchase_count],
        color=["#20A486", "#126B68"]
    )
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of Records")
    ax.grid(axis="y", alpha=0.2)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            str(int(height)),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center"
        )

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ============================================================
# 13. EVALUATE ALL MODELS
# ============================================================

all_results = {}

for model_name, model in models.items():
    try:
        predictions, scores, metrics = evaluate_model(
            model, X_test, y_test
        )
        all_results[model_name] = {
            "predictions": predictions,
            "scores": scores,
            "metrics": metrics
        }
    except Exception as error:
        st.warning(f"Could not evaluate {model_name}: {error}")

if not all_results:
    st.error("No models could be evaluated.")
    st.stop()


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

st.markdown("---")
st.markdown("## Model Comparison")

comparison_data = {
    name: result["metrics"]
    for name, result in all_results.items()
}
comparison_df = pd.DataFrame(comparison_data).T
comparison_df = comparison_df[METRIC_NAMES]

st.dataframe(
    comparison_df.style.format("{:.4f}", na_rep="N/A"),
    width="stretch"
)


# ============================================================
# 15. BEST MODEL PER METRIC
# ============================================================

st.markdown("### Best Model Per Metric")

best_columns = st.columns(3)

for index, metric in enumerate(METRIC_NAMES):
    valid_scores = comparison_df[metric].dropna()
    if valid_scores.empty:
        continue

    best_model = valid_scores.idxmax()
    best_score = valid_scores.max()

    with best_columns[index % 3]:
        st.metric(
            label=f"Best {metric}",
            value=best_model,
            delta=f"{best_score:.4f}"
        )


# ============================================================
# 16. MODEL PERFORMANCE COMPARISON CHART
# ============================================================

st.markdown("---")
st.markdown("### Model Performance Comparison")

selected_metric = st.selectbox("Select Metric", METRIC_NAMES)
chart_values = comparison_df[selected_metric].dropna().sort_values()

fig, ax = plt.subplots(figsize=(9, 4))
chart_values.plot(kind="barh", ax=ax, color="#168574")
ax.set_xlabel(selected_metric)
ax.set_ylabel("Machine Learning Models")
ax.set_title(f"Model Comparison - {selected_metric}")
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()

st.pyplot(fig, width="stretch")
plt.close(fig)


# ============================================================
# 17. INDIVIDUAL MODEL EVALUATION
# ============================================================

st.markdown("---")
st.markdown("## Individual Model Evaluation")

selected_model = st.selectbox(
    "Select a Model",
    list(all_results.keys())
)

selected_result = all_results[selected_model]
predictions = selected_result["predictions"]
scores = selected_result["scores"]
metrics = selected_result["metrics"]


# ============================================================
# 18. INDIVIDUAL MODEL METRICS
# ============================================================

st.markdown(f"### {selected_model} Performance")
metric_columns = st.columns(3)

for index, metric in enumerate(METRIC_NAMES):
    value = metrics[metric]
    with metric_columns[index % 3]:
        st.metric(
            label=metric,
            value=f"{value:.4f}" if pd.notna(value) else "N/A"
        )


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

st.markdown("---")
st.markdown("### Confusion Matrix")

cm = confusion_matrix(y_test, predictions, labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 4))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Purchase", "Purchase"]
)
display.plot(
    cmap="Greens",
    values_format="d",
    ax=ax,
    colorbar=False
)
ax.set_title(f"Confusion Matrix - {selected_model}")
plt.tight_layout()

st.pyplot(fig, width="content")
plt.close(fig)


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

st.markdown("### Classification Report")

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["No Purchase", "Purchase"],
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format("{:.4f}"),
    width="stretch"
)


# ============================================================
# 21. ROC CURVE COMPARISON
# ============================================================

st.markdown("---")
st.markdown("## ROC Curve Comparison")

fig, ax = plt.subplots(figsize=(8, 5))

for model_name, result in all_results.items():
    model_scores = result["scores"]
    if model_scores is None:
        continue

    fpr, tpr, _ = roc_curve(y_test, model_scores)
    auc_value = result["metrics"]["AUC"]

    ax.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_value:.4f})"
    )

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right")
ax.grid(alpha=0.2)
plt.tight_layout()

st.pyplot(fig, width="stretch")
plt.close(fig)


# ============================================================
# 22. DOWNLOAD RESULTS
# ============================================================

st.markdown("---")
st.markdown("## Download Results")

comparison_csv = comparison_df.to_csv(
    float_format="%.4f"
).encode("utf-8")

st.download_button(
    label="Download Model Comparison Results",
    data=comparison_csv,
    file_name="online_shoppers_model_comparison.csv",
    mime="text/csv",
    width="stretch"
)

prediction_df = data.copy()
prediction_df["Predicted_Revenue"] = predictions
prediction_df["Purchase_Prediction"] = np.where(
    predictions == 1, "Purchase", "No Purchase"
)

prediction_csv = prediction_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Prediction Results",
    data=prediction_csv,
    file_name="online_shoppers_predictions.csv",
    mime="text/csv",
    width="stretch"
)


# ============================================================
# 23. FOOTER
# ============================================================

render_footer()
