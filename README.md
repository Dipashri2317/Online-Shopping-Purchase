# Online Shoppers Purchase Prediction

## M.Tech (AIML) – Machine Learning Assignment 2

**Work Integrated Learning Programmes Division**  
**BITS Pilani**

---

## a. Problem Statement

Online shopping websites generate large amounts of customer browsing and purchasing data. Understanding customer purchasing intention can help businesses identify visitors who are more likely to make a purchase.

The objective of this project is to develop a machine learning-based classification system that predicts whether an online shopping session will result in a purchase based on visitor behaviour and session-related features.

Five different machine learning classification algorithms are implemented and evaluated using the same dataset. An interactive Streamlit web application is developed to demonstrate model performance, compare evaluation metrics and allow users to upload test datasets for prediction and evaluation.

The application uses previously trained and saved models. It does not retrain the models when a user uploads a test dataset.

## b. Dataset Description

**Dataset Name:** UCI Online Shoppers Purchasing Intention Dataset  
**Dataset Source:** UCI Machine Learning Repository  
**Problem Type:** Binary Classification

The Online Shoppers Purchasing Intention Dataset contains browsing behaviour and session-related information collected from online shopping visitors. The dataset is used to predict whether a visitor will generate revenue (make a purchase) during a browsing session.

### Dataset Details

| Description | Details |
|---|---|
| Original number of instances | 12,330 |
| Number of input features | 17 |
| Total columns including target | 18 |
| Target column | `Revenue` |
| Classification type | Binary classification |
| Non-purchase sessions | 10,422 |
| Purchase sessions | 1,908 |

### Target Variable

The target variable is `Revenue`.

- `False` / `0` – No purchase made.
- `True` / `1` – Purchase made.

### Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the Online Shoppers Purchasing Intention dataset.
2. Separated input features and the target variable (`Revenue`).
3. Converted the target variable into binary labels.
4. Identified numerical and categorical features.
5. Applied StandardScaler to numerical features.
6. Applied OneHotEncoder to categorical features.
7. Used a ColumnTransformer to apply the appropriate preprocessing to each feature type.
8. Split the dataset into training and testing sets using an 80:20 split.
9. Used stratified sampling to maintain the target class distribution.
10. Trained and evaluated five machine learning classification models.

The preprocessing steps are integrated into the model pipelines to ensure consistent transformations during training and evaluation.

## c. GitHub Repository Link

**GitHub Repository:**

https://github.com/Dipashri2317/Online-Shopping-Purchase.git

The repository contains:

- Complete Python source code.
- Streamlit application (`app.py`).
- `requirements.txt` containing project dependencies.
- `README.md` containing project documentation.
- `download_dataset.py` for downloading the dataset.
- `train_models.py` for training and saving the five classification models.
- `test_data.csv` used for model evaluation.
- Saved trained models for all five classifiers.
- Model metadata and supporting preprocessing artifacts.
- Modular Python files for data handling, evaluation, charts and application styling.

### Project Structure

```text
online-shoppers-purchase-prediction/
│
├── app.py
├── train_models.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
├── online_shoppers_intention.csv
├── test_data.csv
│
├── components/
│   ├── __init__.py
│   ├── styles.py
│   ├── charts.py
│   └── dashboard.py
│
├── utils/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── data_handler.py
│   └── evaluation.py
│
└── model/
    ├── metadata.json
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── gaussian_naive_bayes.pkl
    └── svm.pkl
```

## d. Models Used and Evaluation Metrics

Five machine learning classification algorithms were implemented using the same processed dataset.

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a record belonging to a particular class. It is used to predict whether an online shopping session will result in a purchase.

### 2. Decision Tree Classifier

Decision Tree is a supervised learning algorithm that makes predictions using a sequence of decision rules based on input features. It divides the dataset into smaller groups to classify purchasing behaviour.

### 3. K-Nearest Neighbor (KNN) Classifier

KNN is a distance-based classification algorithm. It classifies a new data point based on the majority class among its nearest neighbours.

### 4. Gaussian Naive Bayes Classifier

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that numerical features follow a Gaussian distribution within each class.

### 5. Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm that identifies an optimal decision boundary to separate different classes. It aims to maximize the margin between the classes.

### Model Comparison Table

The following table presents the evaluation results obtained from the five trained models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8460 | 0.9097 | 0.5050 | 0.7958 | 0.6179 | 0.5488 |
| Decision Tree | 0.8537 | 0.7086 | 0.5352 | 0.4974 | 0.5156 | 0.4300 |
| KNN | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Gaussian Naive Bayes | 0.2773 | 0.7502 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| SVM | 0.8689 | 0.9095 | 0.5589 | 0.7696 | 0.6476 | 0.5805 |

### Evaluation Metrics

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures how many sessions predicted as purchases actually resulted in purchases.
- **Recall:** Measures how many actual purchase sessions were correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient measures the quality of binary classifications using all four confusion matrix categories.

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved an accuracy of 84.60% and the highest AUC of 0.9097. It provides strong class-separation performance and a good balance between recall and precision. |
| Decision Tree | Achieved an accuracy of 85.37%. However, its AUC of 0.7086 and recall of 0.4974 indicate comparatively weaker performance in identifying purchase sessions. |
| KNN | Achieved the highest accuracy of 87.34% and the highest precision of 0.6622. However, its recall of 0.3901 indicates that it misses a considerable number of actual purchase sessions. |
| Gaussian Naive Bayes | Achieved the highest recall of 0.9791 but the lowest accuracy of 27.73% and precision of 0.1756. It identifies most actual purchase sessions but also produces a large number of false positive predictions. |
| SVM | Achieved an accuracy of 86.89%, an F1 Score of 0.6476 and the highest MCC of 0.5805. It demonstrates the strongest overall balance between precision and recall among the five models. |

### Overall Winner for the Dataset

**Support Vector Machine (SVM)** is considered the strongest overall model based on the balance of evaluation metrics.

It achieved:

- Accuracy: 86.89%
- AUC: 0.9095
- Precision: 0.5589
- Recall: 0.7696
- F1 Score: 0.6476
- MCC: 0.5805

SVM achieved the highest F1 Score and MCC among the five evaluated classifiers, indicating a comparatively strong balance between precision and recall and overall classification quality.

However, KNN achieved the highest accuracy, while Logistic Regression achieved the highest AUC. Therefore, the most suitable model can depend on the business objective and the importance assigned to identifying potential purchases.

## Streamlit Web Application

An interactive web application was developed using Streamlit to demonstrate the trained classification models.

### Application Features

1. **Dataset Upload:** Allows users to upload a CSV test dataset containing the required input features and target column.
2. **Model Evaluation:** Loads the saved trained models and evaluates them on the uploaded test dataset without retraining.
3. **Model Comparison:** Displays the evaluation metrics of all five classification models.
4. **Model Selection:** Provides a dropdown to select and inspect an individual model.
5. **Evaluation Metrics:** Displays Accuracy, AUC, Precision, Recall, F1 Score and MCC.
6. **Confusion Matrix:** Visualizes the classification results for the selected model.
7. **Classification Report:** Displays the classification performance for each class.
8. **ROC Curve:** Compares the ROC curves of the five models.
9. **Data Visualization:** Displays dataset overview and class distribution.
10. **Download Results:** Allows users to download the model evaluation results as a CSV file.
11. **Custom UI Theme:** Provides a gradient-based interface with customized styling.

## Live Streamlit Application

**Streamlit App Link:**

https://online-shopping-purchase-bits.streamlit.app/

The application is deployed using Streamlit Community Cloud.

## How to Run the Application Locally

### Step 1: Clone the GitHub Repository

```bash
git clone https://github.com/Dipashri2317/Online-Shopping-Purchase.git
```

### Step 2: Navigate to the Project Folder

```bash
cd online-shoppers-purchase-prediction
```

### Step 3: Create and Activate a Virtual Environment (Optional)

```bash
python -m venv venv
```

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Download the Dataset

```bash
python download_dataset.py
```

### Step 6: Train the Models

```bash
python train_models.py
```

This step generates the trained model files, model metadata and test dataset.

### Step 7: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the web browser.

## Conclusion

This project demonstrates the application of five machine learning classification algorithms for predicting online shoppers' purchasing intention.

The models were evaluated using six performance metrics: Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient.

The trained models were integrated into an interactive Streamlit application, allowing users to upload test data, compare model performance and visualize classification results.

Based on the current evaluation results, SVM demonstrated the strongest overall balance with the highest F1 Score and MCC, while KNN achieved the highest accuracy and Logistic Regression achieved the highest AUC.

The project highlights how machine learning techniques can be applied to online shopping behaviour analysis to support data-driven business decisions.
