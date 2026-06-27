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

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

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

df = pd.read_csv("Red_Wine_Quality.csv")

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

print("\nQuality Score Counts:")
print(df['quality'].value_counts().sort_index())

# ==========================================
# 4. DATA CLEANING
# ==========================================

# Remove duplicates

df.drop_duplicates(inplace=True)

# ==========================================
# 5. DETAILED EDA
# ==========================================

# Quality score distribution

plt.figure(figsize=(7,5))
sns.countplot(
    x='quality',
    data=df
)
plt.title("Wine Quality Score Distribution")
plt.show()

# ------------------------------------------

# Correlation Heatmap

plt.figure(figsize=(12,10))
sns.heatmap(
    df.corr(),
    annot=True,
    fmt='.2f',
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------

# Alcohol vs Quality

plt.figure(figsize=(8,5))
sns.boxplot(
    x='quality',
    y='alcohol',
    data=df
)
plt.title("Alcohol Content vs Quality Score")
plt.show()

# ------------------------------------------

# Volatile Acidity vs Quality

plt.figure(figsize=(8,5))
sns.boxplot(
    x='quality',
    y='volatile acidity',
    data=df
)
plt.title("Volatile Acidity vs Quality Score")
plt.show()

# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================

# Turn the 3-8 quality score into a binary label:
# good (1) if quality >= 7, not good (0) otherwise

df['good_quality'] = (df['quality'] >= 7).astype(int)

print("\nGood vs Not Good Wine Counts:")
print(df['good_quality'].value_counts())

X = df.drop(
    ['quality', 'good_quality'],
    axis=1
)

y = df['good_quality']

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
# 9. FINDING THE BEST K
# ==========================================

k_values = range(1, 31)
accuracies = []

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    accuracies.append(knn_k.score(X_test, y_test))

best_k = k_values[np.argmax(accuracies)]

print("\nBest K:", best_k)
print("Best Accuracy at that K:", max(accuracies))

plt.figure(figsize=(8,5))
plt.plot(
    k_values,
    accuracies,
    marker='o'
)
plt.xlabel("K (number of neighbors)")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs K")
plt.show()

# ==========================================
# 10. KNN MODEL
# ==========================================

model = KNeighborsClassifier(n_neighbors=best_k)

model.fit(
    X_train,
    y_train
)

# ==========================================
# 11. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1]

# ==========================================
# 12. EVALUATION METRICS
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
# 13. OVERFITTING CHECK
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
# 14. CROSS VALIDATION
# ==========================================

cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=best_k),
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
# 15. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['not good', 'good'],
    yticklabels=['not good', 'good']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# 16. ROC CURVE
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
