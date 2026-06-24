# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv("breast-cancer.csv")

print("First 5 Rows")
print(df.head())

# ==========================================
# 3. QUICK EDA
# ==========================================

print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nStatistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nClass Balance (diagnosis):")
print(df['diagnosis'].value_counts())

# ==========================================
# 4. DATA CLEANING
# ==========================================

# Drop ID column

df.drop("id", axis=1, inplace=True)

# Remove duplicates

df.drop_duplicates(inplace=True)

# Encode target: M (malignant) -> 1, B (benign) -> 0

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

print("\nEncoding map:", dict(zip(le.classes_, le.transform(le.classes_))))

# ==========================================
# 5. DETAILED EDA
# ==========================================

# Class balance countplot

plt.figure(figsize=(6,5))
sns.countplot(
    x='diagnosis',
    data=df
)
plt.title("Class Balance (0 = benign, 1 = malignant)")
plt.show()

# ------------------------------------------

# Correlation Heatmap

plt.figure(figsize=(14,12))
sns.heatmap(
    df.corr(),
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------

# Radius Mean vs Diagnosis

plt.figure(figsize=(8,5))
sns.boxplot(
    x='diagnosis',
    y='radius_mean',
    data=df
)
plt.title("Mean Radius vs Diagnosis")
plt.show()

# ------------------------------------------

# Texture Mean vs Diagnosis

plt.figure(figsize=(8,5))
sns.boxplot(
    x='diagnosis',
    y='texture_mean',
    data=df
)
plt.title("Mean Texture vs Diagnosis")
plt.show()

# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================

X = df.drop(
    'diagnosis',
    axis=1
)

y = df['diagnosis']

# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 8. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================
# 9. LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(max_iter=10000)

model.fit(
    X_train,
    y_train
)

# ==========================================
# 10. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1]

# ==========================================
# 11. EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_proba)

print("\nModel Performance")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", roc_auc)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# 12. OVERFITTING CHECK
# ==========================================

train_acc = model.score(
    X_train,
    y_train
)

test_acc = model.score(
    X_test,
    y_test
)

print("\nOverfitting Check")

print("Train Accuracy:", train_acc)
print("Test Accuracy :", test_acc)

# ==========================================
# 13. CROSS VALIDATION
# ==========================================

cv_scores = cross_val_score(
    LogisticRegression(max_iter=10000),
    scaler.fit_transform(X),
    y,
    cv=5,
    scoring='accuracy'
)

print("\nCross Validation Scores")
print(cv_scores)

print("\nAverage CV Score")
print(cv_scores.mean())

# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['benign', 'malignant'],
    yticklabels=['benign', 'malignant']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# 15. ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8,5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    color='red'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")
plt.legend()

plt.show()

# ==========================================
# 16. FEATURE IMPORTANCE
# ==========================================

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

coefficients = coefficients.sort_values(
    by='Coefficient',
    key=abs,
    ascending=False
)

print("\nFeature Importance")
print(coefficients)

# ==========================================
# 17. FEATURE IMPORTANCE VISUALIZATION
# ==========================================

plt.figure(figsize=(10,8))

sns.barplot(
    x='Coefficient',
    y='Feature',
    data=coefficients.head(10)
)

plt.title("Top 10 Feature Importance")

plt.show()
