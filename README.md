# Obesity Level Prediction using Logistic Regression

## Overview

This project applies machine learning techniques to predict obesity levels based on lifestyle and physical attributes. The dataset is preprocessed using normalization and encoding methods, and two multiclass classification strategies are evaluated:

* One-vs-Rest (OvR)
* One-vs-One (OvO)

Both approaches use Logistic Regression as the base classifier.

---

## Dataset

The dataset is loaded directly from an online source and contains:

* Numerical features (e.g., age, height, weight)
* Categorical features (e.g., eating habits, physical activity)
* Target variable: `NObeyesdad` (obesity level classification)

---

## Data Preprocessing

### 1. Handling Numerical Features

* Numerical columns are identified using `float64` dtype.
* Standardization is applied using `StandardScaler`:

  * Mean = 0
  * Standard deviation = 1

### 2. Handling Categorical Features

* Categorical variables are identified using `object` dtype.
* Target column (`NObeyesdad`) is excluded from encoding.
* One-Hot Encoding is applied using `OneHotEncoder`:

  * `drop='first'` avoids multicollinearity.

### 3. Target Encoding

* The target variable is converted into numerical categories using `.cat.codes`.

---

## Feature and Target Separation

* Features (`X`): All columns except `NObeyesdad`
* Target (`y`): Encoded obesity levels

---

## Train-Test Split

* Data is split into training and testing sets:

  * Training set: 67%
  * Test set: 33%
* `random_state=42` ensures reproducibility

---

## Models

### 1. One-vs-Rest (OvR)

* Uses `LogisticRegression` with:

  * `multi_class='ovr'`
  * `max_iter=1000`
* Trains one classifier per class against all others

### 2. One-vs-One (OvO)

* Uses `OneVsOneClassifier` with Logistic Regression
* Trains one classifier for every pair of classes

---

## Evaluation

* Predictions are made on the test set
* Performance is measured using accuracy score

```python
print(accuracy_score(ytest, y_pred))     # OvR accuracy
print(accuracy_score(ytest, yovo_pred))  # OvO accuracy
```

---

## Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

---

## Key Concepts

* Data normalization
* One-hot encoding
* Multiclass classification strategies
* Logistic regression
* Model evaluation using accuracy

---

## Possible Improvements

* Hyperparameter tuning (e.g., regularization strength)
* Cross-validation
* Trying other classifiers (Random Forest, SVM, Gradient Boosting)
* Feature selection or dimensionality reduction
* Handling class imbalance if present

---