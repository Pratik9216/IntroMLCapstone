"""
Model 1: Linear Regression (OLS)
Classical ML Algorithm for House Price Prediction

This script implements a baseline Linear Regression model with:
- Proper preprocessing (scaling + encoding)
- 5-fold cross-validation
- Hyperparameter tuning (regularization)
- Performance evaluation (MSE, MAE, R²)
- Visualizations (actual vs predicted, residuals)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL 1: LINEAR REGRESSION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading preprocessed data...")
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Load feature info
with open('data/feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)
    numerical_features = feature_info['numerical_features']
    categorical_features = feature_info['categorical_features']

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# ============================================================================
# 2. CREATE PREPROCESSING PIPELINE
# ============================================================================
print("\n2. Creating preprocessing pipeline...")

# Numerical pipeline (with scaling - important for Linear Regression)
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combined preprocessor
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

print("✓ Preprocessing pipeline created")

# ============================================================================
# 3. BASELINE LINEAR REGRESSION
# ============================================================================
print("\n3. Training baseline Linear Regression...")

# Create pipeline
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit on training data
lr_pipeline.fit(X_train, y_train)

# Predictions
y_train_pred = lr_pipeline.predict(X_train)
y_test_pred = lr_pipeline.predict(X_test)

# Evaluate
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nBaseline Linear Regression Results:")
print(f"Training   - MSE: {train_mse:,.2f}, MAE: {train_mae:,.2f}, R²: {train_r2:.4f}")
print(f"Test       - MSE: {test_mse:,.2f}, MAE: {test_mae:,.2f}, R²: {test_r2:.4f}")

# ============================================================================
# 4. HYPERPARAMETER TUNING (Ridge Regression)
# ============================================================================
print("\n4. Hyperparameter tuning with Ridge Regression...")

# Ridge regression with different alpha values
ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

# Parameter grid
param_grid = {
    'regressor__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
}

# Grid search with 5-fold CV
grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV MSE: {-grid_search.best_score_:,.2f}")

# Best model
best_model = grid_search.best_estimator_

# Predictions with best model
y_train_pred_best = best_model.predict(X_train)
y_test_pred_best = best_model.predict(X_test)

# Evaluate
train_mse_best = mean_squared_error(y_train, y_train_pred_best)
train_mae_best = mean_absolute_error(y_train, y_train_pred_best)
train_r2_best = r2_score(y_train, y_train_pred_best)

test_mse_best = mean_squared_error(y_test, y_test_pred_best)
test_mae_best = mean_absolute_error(y_test, y_test_pred_best)
test_r2_best = r2_score(y_test, y_test_pred_best)

print(f"\nBest Ridge Regression Results:")
print(f"Training   - MSE: {train_mse_best:,.2f}, MAE: {train_mae_best:,.2f}, R²: {train_r2_best:.4f}")
print(f"Test       - MSE: {test_mse_best:,.2f}, MAE: {test_mae_best:,.2f}, R²: {test_r2_best:.4f}")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n5. Generating visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Actual vs Predicted (Training)
axes[0, 0].scatter(y_train, y_train_pred_best, alpha=0.5, s=20)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price', fontsize=12)
axes[0, 0].set_ylabel('Predicted Price', fontsize=12)
axes[0, 0].set_title('Linear Regression: Actual vs Predicted (Training)', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# 2. Actual vs Predicted (Test)
axes[0, 1].scatter(y_test, y_test_pred_best, alpha=0.5, s=20, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Price', fontsize=12)
axes[0, 1].set_ylabel('Predicted Price', fontsize=12)
axes[0, 1].set_title('Linear Regression: Actual vs Predicted (Test)', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals Histogram (Test)
residuals_test = y_test - y_test_pred_best
axes[1, 0].hist(residuals_test, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Residuals', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Linear Regression: Residuals Distribution (Test)', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# 4. Residuals vs Predicted (Test)
axes[1, 1].scatter(y_test_pred_best, residuals_test, alpha=0.5, s=20, color='purple')
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Predicted Price', fontsize=12)
axes[1, 1].set_ylabel('Residuals', fontsize=12)
axes[1, 1].set_title('Linear Regression: Residuals vs Predicted (Test)', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/01_linear_regression.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/01_linear_regression.png")
plt.close()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n6. Saving results...")

results = {
    'Model': 'Linear Regression (Ridge)',
    'Best_Alpha': grid_search.best_params_['regressor__alpha'],
    'Train_MSE': train_mse_best,
    'Train_MAE': train_mae_best,
    'Train_R2': train_r2_best,
    'Test_MSE': test_mse_best,
    'Test_MAE': test_mae_best,
    'Test_R2': test_r2_best
}

# Save to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results/01_linear_regression_results.csv', index=False)
print("✓ Saved: results/01_linear_regression_results.csv")

# Save model
with open('results/01_linear_regression_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Saved: results/01_linear_regression_model.pkl")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("LINEAR REGRESSION - SUMMARY")
print("="*80)
print(f"Best Hyperparameter: alpha = {grid_search.best_params_['regressor__alpha']}")
print(f"\nTest Set Performance:")
print(f"  MSE: {test_mse_best:,.2f}")
print(f"  MAE: {test_mae_best:,.2f}")
print(f"  R²:  {test_r2_best:.4f}")
print(f"\nKey Insights:")
print(f"  • Linear model provides interpretable baseline")
print(f"  • Ridge regularization (alpha={grid_search.best_params_['regressor__alpha']}) helps prevent overfitting")
print(f"  • Feature scaling is crucial for linear models")
print(f"  • R² of {test_r2_best:.4f} indicates model explains {test_r2_best*100:.1f}% of variance")
print("\n" + "="*80)
