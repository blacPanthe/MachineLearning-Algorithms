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

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv("/content/drive/MyDrive/Real estate.csv")

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

# ==========================================
# 4. DATA CLEANING
# ==========================================

# Drop ID column

df.drop("No", axis=1, inplace=True)

# Remove duplicates

df.drop_duplicates(inplace=True)

# ==========================================
# 5. DETAILED EDA
# ==========================================

# Correlation Heatmap

plt.figure(figsize=(10,6))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------

# House Age vs Price

plt.figure(figsize=(8,5))
sns.scatterplot(
    x='X2 house age',
    y='Y house price of unit area',
    data=df
)
plt.title("House Age vs House Price")
plt.show()

# ------------------------------------------

# MRT Distance vs Price

plt.figure(figsize=(8,5))
sns.scatterplot(
    x='X3 distance to the nearest MRT station',
    y='Y house price of unit area',
    data=df
)
plt.title("MRT Distance vs House Price")
plt.show()

# ------------------------------------------

# Convenience Stores vs Price

plt.figure(figsize=(8,5))
sns.boxplot(
    x='X4 number of convenience stores',
    y='Y house price of unit area',
    data=df
)
plt.title("Convenience Stores vs House Price")
plt.show()

# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================

X = df.drop(
    'Y house price of unit area',
    axis=1
)

y = df['Y house price of unit area']

# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 8. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================
# 9. LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# ==========================================
# 10. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 11. EVALUATION METRICS
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nModel Performance")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

# ==========================================
# 12. OVERFITTING CHECK
# ==========================================

train_r2 = model.score(
    X_train,
    y_train
)

test_r2 = model.score(
    X_test,
    y_test
)

print("\nOverfitting Check")

print("Train R²:", train_r2)
print("Test R² :", test_r2)

# ==========================================
# 13. CROSS VALIDATION
# ==========================================

cv_scores = cross_val_score(
    LinearRegression(),
    scaler.fit_transform(X),
    y,
    cv=5,
    scoring='r2'
)

print("\nCross Validation Scores")
print(cv_scores)

print("\nAverage CV Score")
print(cv_scores.mean())

# ==========================================
# 14. ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(8,5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices")

plt.show()

# ==========================================
# 15. RESIDUAL ANALYSIS
# ==========================================

residuals = y_test - y_pred

# Residual Plot

plt.figure(figsize=(8,5))

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0,
    color='red',
    linestyle='--'
)

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.show()

# ==========================================
# 16. RESIDUAL DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(
    residuals,
    kde=True
)

plt.title("Residual Distribution")

plt.show()

# ==========================================
# 17. FEATURE IMPORTANCE
# ==========================================

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

coefficients = coefficients.sort_values(
    by='Coefficient',
    key=abs,
    ascending=False
)

print("\nFeature Importance")
print(coefficients)

# ==========================================
# 18. FEATURE IMPORTANCE VISUALIZATION
# ==========================================

plt.figure(figsize=(10,6))

sns.barplot(
    x='Coefficient',
    y='Feature',
    data=coefficients
)

plt.title("Feature Importance")

plt.show()