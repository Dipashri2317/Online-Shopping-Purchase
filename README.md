# Phishing Website Detection

## M.Tech (AIML) – Machine Learning Assignment 2

**Work Integrated Learning Programmes Division**
**BITS Pilani**

---

## a. Problem Statement

Phishing websites are fraudulent websites designed to imitate legitimate websites and deceive users into revealing sensitive information such as login credentials, banking details and personal information.

The objective of this project is to develop a machine learning-based classification system that identifies whether a website is legitimate or phishing based on its website-related features.

Five different machine learning classification algorithms are implemented and evaluated on the same dataset. An interactive Streamlit web application is developed to demonstrate model performance, compare evaluation metrics and allow users to upload test datasets for prediction and evaluation.

The application uses previously trained and saved models. It does not retrain the models when a user uploads a test dataset.

## b. Dataset Description

**Dataset Name:** UCI Phishing Websites Dataset
**Dataset Source:** UCI Machine Learning Repository
**Problem Type:** Binary Classification

The UCI Phishing Websites Dataset contains website-related attributes that help distinguish legitimate websites from phishing websites.

### Dataset Details

| Description                                 | Details               |
| ------------------------------------------- | --------------------- |
| Original number of instances                | 11,055                |
| Number of instances after duplicate removal | 5,849                 |
| Number of input features                    | 30                    |
| Target column                               | `is_phishing`         |
| Classification type                         | Binary classification |
| Legitimate records                          | 2,830                 |
| Phishing records                            | 3,019                 |

### Target Variable

The target variable is `is_phishing`.

- `0` – Legitimate website
- `1` – Phishing website

### Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset from the ARFF file.
2. Converted the dataset into a suitable tabular format.
3. Converted the feature and target columns into numeric format.
4. Removed duplicate records.
5. Mapped the target variable to binary labels.
6. Split the processed dataset into training and testing sets.

The processed dataset contains 5,849 records and 30 input features.

## c. GitHub Repository Link

**GitHub Repository:**
[PASTE YOUR GITHUB REPOSITORY LINK HERE]

The repository contains:

- Complete Python source code.
- Streamlit application (`app.py`).
- `requirements.txt` containing project dependencies.
- `README.md` containing project documentation.
- `test_data.csv` used for model evaluation.
- Saved trained models for all five classifiers.
- Model metadata and supporting preprocessing artifacts.
- Modular Python files for data handling, evaluation, charts and application styling.

### Project Structure

```text
phishing/
│
├── app.py
├── requirements.txt
├── README.md
├── online_shoppers_intention.csv
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
└── models/
    ├── metadata.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_naive_bayes.joblib
    └── svm.joblib
```

## d. Models Used and Evaluation Metrics

Five machine learning classification algorithms were implemented using the same processed dataset.

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a record belonging to a particular class. It is used as a baseline classification model for distinguishing legitimate and phishing websites.

### 2. Decision Tree Classifier

Decision Tree is a supervised learning algorithm that makes predictions using a sequence of decision rules based on input features. It divides the dataset into smaller groups to classify websites.

### 3. K-Nearest Neighbor (KNN) Classifier

KNN is a distance-based classification algorithm. It classifies a new data point based on the majority class among its nearest neighbours.

### 4. Gaussian Naive Bayes Classifier

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that the features follow a Gaussian distribution within each class.

### 5. Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm that identifies an optimal decision boundary to separate different classes. It aims to maximize the margin between the classes.

### Model Comparison Table

The following table presents the evaluation results obtained from the trained models.

| ML Model Name          | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ---------------------- | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression    |   0.9120 | 0.9722 |    0.9168 | 0.9024 |   0.9096 | 0.8239 |
| Decision Tree          |   0.9291 | 0.9398 |    0.9392 | 0.9146 |   0.9267 | 0.8583 |
| KNN                    |   0.9265 | 0.9817 |    0.9136 | 0.9390 |   0.9261 | 0.8533 |
| Gaussian Naive Bayes   |   0.6256 | 0.9653 |    0.5673 | 0.9983 |   0.7235 | 0.3856 |
| Support Vector Machine |   0.9299 | 0.9831 |    0.9473 | 0.9077 |   0.9270 | 0.8604 |

**Evaluation Metrics:**

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures how many predicted phishing or legitimate classifications are correct.
- **Recall:** Measures how many actual instances of a class are correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **MCC:** Measures the quality of binary classifications using all four confusion matrix categories.

### Observations on Model Performance

| ML Model Name          | Observation about Model Performance                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression    | Achieved an accuracy of 91.20% and an AUC score of 0.9722. It provides a strong baseline for classifying phishing and legitimate websites.              |
| Decision Tree          | Achieved an accuracy of 92.91%. It performed well in terms of classification accuracy, although its AUC score was lower than those of the other models. |
| KNN                    | Achieved an accuracy of 92.65% and an AUC score of 0.9817. It demonstrated strong classification and class-separation performance.                      |
| Gaussian Naive Bayes   | Achieved an accuracy of 62.56% and an AUC score of 0.9653. Its classification accuracy was comparatively lower, despite a relatively high AUC score.    |
| Support Vector Machine | Achieved the highest accuracy of 92.99% and the highest AUC score of 0.9831 among the five evaluated models.                                            |

### Overall Winner for the Dataset

**Support Vector Machine (SVM)** is the overall best-performing model based on the current evaluation results. It achieved the highest accuracy (92.99%) and AUC score (0.9831) among the five classifiers.

However, the final model comparison should also consider Precision, Recall, F1 Score and MCC.

## Streamlit Web Application

An interactive web application was developed using Streamlit to demonstrate the trained classification models.

### Application Features

1. **Dataset Upload:** Allows users to upload a CSV test dataset containing all required input features and the target column.
2. **Model Evaluation:** Loads the saved trained models and evaluates them on the uploaded test dataset without retraining.
3. **Model Comparison:** Displays the evaluation metrics of all five classification models.
4. **Model Selection:** Provides a dropdown to select and inspect an individual model.
5. **Evaluation Metrics:** Displays Accuracy, AUC, Precision, Recall, F1 Score and MCC.
6. **Confusion Matrix:** Visualizes the classification results for the selected model.
7. **Classification Report:** Displays the classification performance for each class.
8. **ROC Curve:** Compares the ROC curves of the five models.
9. **Data Visualization:** Displays dataset overview and class distribution.
10. **Download Results:** Allows users to download the model evaluation results as a CSV file.

## Live Streamlit Application

**Streamlit App Link:**
[PASTE YOUR DEPLOYED STREAMLIT APP LINK HERE]

The application is deployed using Streamlit Community Cloud.

## How to Run the Application Locally

### Step 1: Clone the GitHub Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### Step 2: Navigate to the Project Folder

```bash
cd phishing
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the web browser.

## Conclusion

This project demonstrates the application of five machine learning classification algorithms for phishing website detection.

The models were evaluated using six performance metrics: Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient.

The trained models were integrated into an interactive Streamlit application, allowing users to upload test data, compare model performance and visualize classification results.

Based on the current evaluation results, Support Vector Machine achieved the best overall performance among the five evaluated classifiers.

