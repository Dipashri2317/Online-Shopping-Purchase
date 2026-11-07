import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from components.styles import (
    apply_custom_styles,
    render_header,
    render_footer
)

from components.charts import (
    create_class_distribution_chart,
    create_model_comparison_chart,
    create_confusion_matrix_chart,
    create_roc_curve_chart
)

from components.dashboard import (
    create_classification_report
)

from utils.model_loader import (
    load_models,
    load_metadata
)

from utils.data_handler import (
    load_default_dataset,
    validate_dataset,
    convert_target_values
)

from utils.evaluation import evaluate_model


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(

    page_title="Online Shoppers Purchase Prediction",

    page_icon="🛒",

    layout="wide"
)


# --------------------------------------------------
# APPLY CUSTOM THEME
# --------------------------------------------------

apply_custom_styles()

MODEL_NAMES = [

    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Gaussian Naive Bayes",
    "SVM"
]


METRIC_NAMES = [

    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]


# --------------------------------------------------
# LOAD METADATA
# --------------------------------------------------

metadata = load_metadata()

FEATURE_COLUMNS = metadata["features"]

TARGET_COLUMN = metadata["target_column"]


# --------------------------------------------------
# HEADER
# --------------------------------------------------

render_header()


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

models = load_models()


if not models:

    st.error(
        "No trained models were found. "
        "Run `python train_models.py` first."
    )

    st.stop()


# --------------------------------------------------
# LOAD DEFAULT DATASET
# --------------------------------------------------

try:

    sample_dataset = load_default_dataset()

except Exception:

    sample_dataset = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Purchase Prediction")

st.sidebar.markdown(
    """
    Upload a labelled CSV file to evaluate
    the saved machine learning models.
    """
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

    - CSV format
    - All 17 input features
    - One target column
    - Revenue values: True/False or 0/1
    - Both target classes must be present
    """
)


st.sidebar.markdown("---")

st.sidebar.subheader("Sample Dataset")


if sample_dataset is not None:

    st.sidebar.download_button(

        label="Download Sample Dataset",

        data=sample_dataset.to_csv(
            index=False
        ).encode("utf-8"),

        file_name="test_data.csv",

        mime="text/csv",

        width="stretch"
    )


# --------------------------------------------------
# DATASET SELECTION
# --------------------------------------------------

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        dataset_source = "Uploaded Dataset"

    except Exception as error:

        st.error(
            f"Unable to read uploaded CSV file: {error}"
        )

        st.stop()

    target_options = [

        column
        for column in data.columns
        if column not in FEATURE_COLUMNS
    ]

    if not target_options:

        st.error(
            "No target column found. "
            "Include a Revenue/target column."
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

    if selected_target != TARGET_COLUMN:

        data = data.rename(

            columns={
                selected_target: TARGET_COLUMN
            }
        )

else:

    try:

        data = load_default_dataset()

        dataset_source = "Default Test Dataset"

    except Exception as error:

        st.error(
            f"Unable to load the default test dataset: {error}"
        )

        st.info(
            "Run `python train_models.py` "
            "after placing the source CSV in this folder."
        )

        st.stop()


# --------------------------------------------------
# VALIDATE DATASET
# --------------------------------------------------

is_valid, validation_message, data = validate_dataset(

    data,

    FEATURE_COLUMNS,

    TARGET_COLUMN
)


if not is_valid:

    st.error(validation_message)

    st.stop()


st.success(
    f"{dataset_source} loaded successfully."
)


X_test = data[FEATURE_COLUMNS]

y_test = convert_target_values(
    data[TARGET_COLUMN]
)


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.markdown("## Dataset Overview")


left_column, right_column = st.columns([1, 1])


total_records = len(data)

purchase_count = int(
    (y_test == 1).sum()
)

no_purchase_count = int(
    (y_test == 0).sum()
)


with left_column:

    st.metric(
        "Total Records",
        f"{total_records:,}"
    )

    st.metric(
        "Input Features",
        len(FEATURE_COLUMNS)
    )

    st.markdown("### Dataset Preview")

    st.dataframe(
        data.head(),
        width="stretch"
    )


with right_column:

    st.markdown(
        "### Purchase Intention Distribution"
    )

    fig = create_class_distribution_chart(

        no_purchase_count,

        purchase_count
    )

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

st.markdown("---")

st.markdown("## Model Evaluation")


all_results = {}


for model_name in MODEL_NAMES:

    if model_name not in models:

        continue

    try:

        predictions, scores, metrics = evaluate_model(

            models[model_name],

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

    st.error(
        "No models could be evaluated. "
        "Check model files and feature preprocessing."
    )

    st.stop()


# --------------------------------------------------
# MODEL COMPARISON TABLE
# --------------------------------------------------

st.markdown("### Model Comparison")


comparison_df = pd.DataFrame(

    {
        name: result["metrics"]

        for name, result in all_results.items()
    }

).T


comparison_df = comparison_df[METRIC_NAMES]


st.dataframe(

    comparison_df.style.format(
        "{:.4f}",
        na_rep="N/A"
    ),

    width="stretch"
)


# --------------------------------------------------
# BEST MODEL PER METRIC
# --------------------------------------------------

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

            f"Best {metric}",

            best_model,

            delta=f"{best_score:.4f}"
        )


# --------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "### Model Performance Comparison"
)


selected_metric = st.selectbox(

    "Select Metric",

    METRIC_NAMES
)


chart_values = (

    comparison_df[selected_metric]

    .dropna()

    .sort_values()
)


fig = create_model_comparison_chart(

    chart_values,

    selected_metric
)


st.pyplot(
    fig,
    width="stretch"
)

plt.close(fig)


# --------------------------------------------------
# INDIVIDUAL MODEL EVALUATION
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "## Individual Model Evaluation"
)


selected_model = st.selectbox(

    "Select a Model",

    list(all_results.keys())
)


selected_result = all_results[selected_model]


predictions = selected_result["predictions"]

scores = selected_result["scores"]

metrics = selected_result["metrics"]


st.markdown(
    f"### {selected_model} Performance"
)


metric_columns = st.columns(3)


for index, metric in enumerate(METRIC_NAMES):

    value = metrics[metric]

    with metric_columns[index % 3]:

        st.metric(

            metric,

            f"{value:.4f}"

            if pd.notna(value)

            else "N/A"
        )


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.markdown("---")

st.markdown("### Confusion Matrix")


fig = create_confusion_matrix_chart(

    y_test,

    predictions
)


st.pyplot(
    fig,
    width="content"
)

plt.close(fig)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.markdown("### Classification Report")


report_df = create_classification_report(

    y_test,

    predictions
)


st.dataframe(

    report_df.style.format(
        "{:.4f}",
        na_rep="N/A"
    ),

    width="stretch"
)


# --------------------------------------------------
# ROC CURVE COMPARISON
# --------------------------------------------------

st.markdown("---")

st.markdown("## ROC Curve Comparison")


fig = create_roc_curve_chart(

    all_results,

    y_test
)


st.pyplot(
    fig,
    width="stretch"
)

plt.close(fig)


# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

st.markdown("---")

st.markdown("## Download Results")


csv_data = comparison_df.to_csv(
    float_format="%.4f"
).encode("utf-8")


st.download_button(

    "Download Model Comparison Results",

    data=csv_data,

    file_name="model_comparison_results.csv",

    mime="text/csv",

    width="stretch"
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

render_footer()