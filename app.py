
# ============================================================
# ONLINE SHOPPERS PURCHASE PREDICTION
# Main Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve
)

from config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    METRIC_NAMES
)

from utils.styles import (
    apply_custom_styles,
    render_header,
    render_footer
)

from utils.data_loader import (
    convert_target_values,
    load_default_dataset,
    create_sample_dataset,
    validate_dataset
)

from utils.model_utils import (
    load_models,
    evaluate_model
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
# 2. APPLY CUSTOM STYLES AND HEADER
# ============================================================

apply_custom_styles()
render_header()


# ============================================================
# 3. LOAD SAVED MODELS
# ============================================================

models = load_models()

if not models:
    st.error(
        "No trained models found. Check that all five .pkl "
        "files are inside the model folder."
    )
    st.stop()


# ============================================================
# 4. SIDEBAR
# ============================================================

st.sidebar.title("Online Shopping Prediction")

st.sidebar.markdown(
    "Upload a CSV file to evaluate the trained "
    "machine learning models."
)

st.sidebar.markdown("---")

st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

st.sidebar.markdown(
    """
    **CSV Requirements**

    - File format: CSV
    - Must contain all 17 input features
    - Target column can have any name
    - Target values must be 0 or 1
    - Must contain both target classes
    """
)

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
    st.sidebar.error(
        f"Sample dataset unavailable: {error}"
    )


# ============================================================
# 5. LOAD DEFAULT OR UPLOADED DATASET
# ============================================================

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)
        dataset_source = "Uploaded Dataset"

    except Exception as error:
        st.error(
            f"Unable to read uploaded CSV file: {error}"
        )
        st.stop()

    # Identify possible target columns.
    target_options = [
        col for col in data.columns
        if col not in FEATURE_COLUMNS
    ]

    if not target_options:
        st.error(
            "No target column found. "
            "Include the actual target in your CSV."
        )
        st.stop()

    default_index = (
        target_options.index(TARGET_COLUMN)
        if TARGET_COLUMN in target_options
        else 0
    )

    selected_target = st.sidebar.selectbox(
        "Select Target Column",
        options=target_options,
        index=default_index
    )

    # Rename the selected target column to Revenue.
    if selected_target != TARGET_COLUMN:

        if TARGET_COLUMN in data.columns:
            st.error(
                f"The CSV already contains a '{TARGET_COLUMN}' column. "
                "Remove the extra target column or select Revenue."
            )
            st.stop()

        data = data.rename(
            columns={selected_target: TARGET_COLUMN}
        )

else:

    try:
        data = load_default_dataset()
        dataset_source = "Default Test Dataset"

    except Exception as error:
        st.error(
            f"Unable to load default dataset: {error}"
        )
        st.stop()


# ============================================================
# 6. VALIDATE DATASET
# ============================================================

if TARGET_COLUMN not in data.columns:
    st.error(
        f"Target column '{TARGET_COLUMN}' was not found."
    )
    st.stop()

try:
    data[TARGET_COLUMN] = convert_target_values(
        data[TARGET_COLUMN]
    )

except Exception as error:
    st.error(str(error))
    st.stop()


is_valid, validation_message = validate_dataset(data)

if not is_valid:
    st.error(validation_message)
    st.stop()


# Prepare input and target variables.
X_test = data[FEATURE_COLUMNS].copy()
y_test = data[TARGET_COLUMN].copy()

st.success(
    f"{dataset_source} loaded successfully."
)


# ============================================================
# 7. DATASET OVERVIEW AND CLASS DISTRIBUTION
# ============================================================

st.markdown("## Dataset Overview")

left_column, right_column = st.columns([1, 1])

total_records = len(data)
total_features = len(FEATURE_COLUMNS)

no_purchase_count = int((y_test == 0).sum())
purchase_count = int((y_test == 1).sum())


with left_column:

    st.metric(
        "Total Records",
        total_records
    )

    st.metric(
        "Input Features",
        total_features
    )

    st.markdown("### Dataset Preview")

    st.dataframe(
        data.head(),
        width="stretch"
    )


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
            (
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center"
        )

    plt.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


# ============================================================
# 8. EVALUATE ALL MODELS
# ============================================================

all_results = {}

for model_name, model in models.items():

    try:

        predictions, scores, metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        all_results[model_name] = {
            "predictions": predictions,
            "scores": scores,
            "metrics": metrics
        }

    except Exception as error:

        st.warning(
            f"Could not evaluate {model_name}: {error}"
        )


if not all_results:
    st.error("No models could be evaluated.")
    st.stop()


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

st.markdown("---")
st.markdown("## Model Comparison")

comparison_data = {
    name: result["metrics"]
    for name, result in all_results.items()
}

comparison_df = pd.DataFrame(
    comparison_data
).T

comparison_df = comparison_df[METRIC_NAMES]

st.dataframe(
    comparison_df.style.format(
        "{:.4f}",
        na_rep="N/A"
    ),
    width="stretch"
)


# ============================================================
# 10. BEST MODEL PER METRIC
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
# 11. MODEL PERFORMANCE COMPARISON CHART
# ============================================================

st.markdown("---")
st.markdown("### Model Performance Comparison")

selected_metric = st.selectbox(
    "Select Metric",
    METRIC_NAMES
)

chart_values = (
    comparison_df[selected_metric]
    .dropna()
    .sort_values()
)

fig, ax = plt.subplots(figsize=(9, 4))

chart_values.plot(
    kind="barh",
    ax=ax,
    color="#168574"
)

ax.set_xlabel(selected_metric)
ax.set_ylabel("Machine Learning Models")

ax.set_title(
    f"Model Comparison - {selected_metric}"
)

ax.grid(
    axis="x",
    alpha=0.2
)

plt.tight_layout()

st.pyplot(
    fig,
    width="stretch"
)

plt.close(fig)


# ============================================================
# 12. INDIVIDUAL MODEL EVALUATION
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
# 13. INDIVIDUAL MODEL METRICS
# ============================================================

st.markdown(
    f"### {selected_model} Performance"
)

metric_columns = st.columns(3)

for index, metric in enumerate(METRIC_NAMES):

    value = metrics[metric]

    with metric_columns[index % 3]:

        st.metric(
            label=metric,
            value=(
                f"{value:.4f}"
                if pd.notna(value)
                else "N/A"
            )
        )


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

st.markdown("---")
st.markdown("### Confusion Matrix")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

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

ax.set_title(
    f"Confusion Matrix - {selected_model}"
)

plt.tight_layout()

st.pyplot(
    fig,
    width="content"
)

plt.close(fig)


# ============================================================
# 15. CLASSIFICATION REPORT
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

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.style.format("{:.4f}"),
    width="stretch"
)


# ============================================================
# 16. ROC CURVE COMPARISON
# ============================================================

st.markdown("---")
st.markdown("## ROC Curve Comparison")

fig, ax = plt.subplots(figsize=(8, 5))

for model_name, result in all_results.items():

    model_scores = result["scores"]

    if model_scores is None:
        continue

    fpr, tpr, _ = roc_curve(
        y_test,
        model_scores
    )

    auc_value = result["metrics"]["AUC"]

    ax.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_value:.4f})"
    )


ax.plot(
    [0, 1],
    [0, 1],
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

st.pyplot(
    fig,
    width="stretch"
)

plt.close(fig)


# ============================================================
# 17. DOWNLOAD RESULTS
# ============================================================

st.markdown("---")
st.markdown("## Download Results")


# Download model comparison results.
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


# Download predictions for the selected model.
prediction_df = data.copy()

prediction_df["Predicted_Revenue"] = predictions

prediction_df["Purchase_Prediction"] = np.where(
    predictions == 1,
    "Purchase",
    "No Purchase"
)

prediction_csv = prediction_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Prediction Results",
    data=prediction_csv,
    file_name="online_shoppers_predictions.csv",
    mime="text/csv",
    width="stretch"
)


# ============================================================
# 18. FOOTER
# ============================================================

render_footer()
