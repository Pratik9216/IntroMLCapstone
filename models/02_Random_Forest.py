"""
Model 2: Random Forest Regressor
Classical ML Algorithm for House Price Prediction

This script implements a Random Forest ensemble model with:
- No scaling needed (tree-based model)
- Hyperparameter tuning (n_estimators, max_depth, min_samples_split)
- Feature importance analysis
- 5-fold cross-validation
- Performance evaluation (MSE, MAE, R²)
- Visualizations (actual vs predicted, residuals, feature importance)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL 2: RANDOM FOREST REGRESSOR")
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

# ============================================================================
# 2. CREATE PREPROCESSING PIPELINE (NO SCALING)
# ============================================================================
print("\n2. Creating preprocessing pipeline...")

# Numerical pipeline (NO scaling for tree-based models)
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
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

print("✓ Preprocessing pipeline created (no scaling for Random Forest)")

# ============================================================================
# 3. BASELINE RANDOM FOREST
# ============================================================================
print("\n3. Training baseline Random Forest...")

# Create pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])

# Fit on training data
rf_pipeline.fit(X_train, y_train)

# Predictions
y_train_pred = rf_pipeline.predict(X_train)
y_test_pred = rf_pipeline.predict(X_test)

# Evaluate
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nBaseline Random Forest Results:")
print(f"Training   - MSE: {train_mse:,.2f}, MAE: {train_mae:,.2f}, R²: {train_r2:.4f}")
print(f"Test       - MSE: {test_mse:,.2f}, MAE: {test_mae:,.2f}, R²: {test_r2:.4f}")

# ============================================================================
# 4. HYPERPARAMETER TUNING
# ============================================================================
print("\n4. Hyperparameter tuning with RandomizedSearchCV...")

# Create new pipeline for tuning
rf_pipeline_tuned = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])

# Parameter distribution for random search
param_distributions = {
    'regressor__n_estimators': [100, 200, 300, 500, 800],
    'regressor__max_depth': [10, 15, 20, 25, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2', None]
}

# Randomized search with 5-fold CV (faster than GridSearch)
random_search = RandomizedSearchCV(
    rf_pipeline_tuned,
    param_distributions,
    n_iter=20,  # Number of parameter settings sampled
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print("Running randomized search (this may take a few minutes)...")
random_search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best CV MSE: {-random_search.best_score_:,.2f}")

# Best model
best_model = random_search.best_estimator_

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

print(f"\nBest Random Forest Results:")
print(f"Training   - MSE: {train_mse_best:,.2f}, MAE: {train_mae_best:,.2f}, R²: {train_r2_best:.4f}")
print(f"Test       - MSE: {test_mse_best:,.2f}, MAE: {test_mae_best:,.2f}, R²: {test_r2_best:.4f}")

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================
print("\n5. Analyzing feature importance...")

# Get feature names after preprocessing
feature_names = (numerical_features + 
                 list(best_model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features)))

# Get feature importances
importances = best_model.named_steps['regressor'].feature_importances_

# Create dataframe
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Top 20 features
top_20_features = feature_importance_df.head(20)
print("\nTop 20 Most Important Features:")
print(top_20_features.to_string(index=False))

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n6. Generating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Actual vs Predicted (Training)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_train, y_train_pred_best, alpha=0.5, s=20)
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Price', fontsize=12)
ax1.set_ylabel('Predicted Price', fontsize=12)
ax1.set_title('Random Forest: Actual vs Predicted (Training)', fontsize=14)
ax1.grid(True, alpha=0.3)

# 2. Actual vs Predicted (Test)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, y_test_pred_best, alpha=0.5, s=20, color='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Price', fontsize=12)
ax2.set_ylabel('Predicted Price', fontsize=12)
ax2.set_title('Random Forest: Actual vs Predicted (Test)', fontsize=14)
ax2.grid(True, alpha=0.3)

# 3. Residuals Histogram (Test)
ax3 = fig.add_subplot(gs[1, 0])
residuals_test = y_test - y_test_pred_best
ax3.hist(residuals_test, bins=30, edgecolor='black', alpha=0.7)
ax3.axvline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Residuals', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Random Forest: Residuals Distribution (Test)', fontsize=14)
ax3.grid(True, alpha=0.3)

# 4. Residuals vs Predicted (Test)
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_test_pred_best, residuals_test, alpha=0.5, s=20, color='purple')
ax4.axhline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Price', fontsize=12)
ax4.set_ylabel('Residuals', fontsize=12)
ax4.set_title('Random Forest: Residuals vs Predicted (Test)', fontsize=14)
ax4.grid(True, alpha=0.3)

# 5. Feature Importance (Top 20)
ax5 = fig.add_subplot(gs[2, :])
ax5.barh(range(20), top_20_features['Importance'].values)
ax5.set_yticks(range(20))
ax5.set_yticklabels(top_20_features['Feature'].values, fontsize=10)
ax5.set_xlabel('Importance', fontsize=12)
ax5.set_title('Random Forest: Top 20 Feature Importances', fontsize=14)
ax5.grid(True, alpha=0.3, axis='x')
ax5.invert_yaxis()

plt.savefig('results/figures/02_random_forest.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/02_random_forest.png")
plt.close()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n7. Saving results...")

results = {
    'Model': 'Random Forest',
    'n_estimators': random_search.best_params_['regressor__n_estimators'],
    'max_depth': random_search.best_params_['regressor__max_depth'],
    'min_samples_split': random_search.best_params_['regressor__min_samples_split'],
    'min_samples_leaf': random_search.best_params_['regressor__min_samples_leaf'],
    'max_features': random_search.best_params_['regressor__max_features'],
    'Train_MSE': train_mse_best,
    'Train_MAE': train_mae_best,
    'Train_R2': train_r2_best,
    'Test_MSE': test_mse_best,
    'Test_MAE': test_mae_best,
    'Test_R2': test_r2_best
}

# Save to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results/02_random_forest_results.csv', index=False)
print("✓ Saved: results/02_random_forest_results.csv")

# Save feature importance
feature_importance_df.to_csv('results/02_random_forest_feature_importance.csv', index=False)
print("✓ Saved: results/02_random_forest_feature_importance.csv")

# Save model
with open('results/02_random_forest_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Saved: results/02_random_forest_model.pkl")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RANDOM FOREST - SUMMARY")
print("="*80)
print(f"Best Hyperparameters:")
print(f"  n_estimators: {random_search.best_params_['regressor__n_estimators']}")
print(f"  max_depth: {random_search.best_params_['regressor__max_depth']}")
print(f"  min_samples_split: {random_search.best_params_['regressor__min_samples_split']}")
print(f"  min_samples_leaf: {random_search.best_params_['regressor__min_samples_leaf']}")
print(f"  max_features: {random_search.best_params_['regressor__max_features']}")
print(f"\nTest Set Performance:")
print(f"  MSE: {test_mse_best:,.2f}")
print(f"  MAE: {test_mae_best:,.2f}")
print(f"  R²:  {test_r2_best:.4f}")
print(f"\nTop 3 Most Important Features:")
for i, row in top_20_features.head(3).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")
print(f"\nKey Insights:")
print(f"  • Ensemble method captures non-linear relationships")
print(f"  • No feature scaling needed (tree-based)")
print(f"  • Feature importance helps interpretability")
print(f"  • R² of {test_r2_best:.4f} indicates strong predictive power")
print("\n" + "="*80)
