import pandas as pd

from sklearn.metrics import classification_report


def create_classification_report(
    y_test,
    predictions
):

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "No Purchase",
            "Purchase"
        ],
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(report).transpose()