from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# ============================================================
# GET MODEL SCORES
# ============================================================

def get_model_scores(model, X):

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return model.predict(X)


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    scores = get_model_scores(model, X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),

        "AUC": roc_auc_score(y_test, scores),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test,
            predictions
        )
    }

    return predictions, scores, metrics