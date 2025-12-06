"""
Paper 1: RailEstate Spatial Feature Engineering
Adapted methodology from RailEstate paper for Kaggle House Prices dataset

This script implements:
1. Neighborhood-based aggregate features (proxy for spatial features)
2. Transportation proximity features from Condition1/Condition2
3. Interaction features between neighborhood and property characteristics
4. XGBoost model with engineered features

Citation: [Add RailEstate paper citation here]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PAPER 1: RAILESTATE SPATIAL FEATURE ENGINEERING")
print("="*80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n1. LOADING PREPROCESSED DATA...")

# Load original training data for feature engineering
df_train_full = pd.read_csv('train.csv')
print(f"Full training data: {df_train_full.shape}")

# Load train/test splits
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# ============================================================================
# 2. NEIGHBORHOOD AGGREGATE FEATURES (Spatial Proxy)
# ============================================================================
print("\n2. CREATING NEIGHBORHOOD AGGREGATE FEATURES...")

# Calculate neighborhood statistics from FULL training data only (prevent data leakage)
neighborhood_stats = df_train_full.groupby('Neighborhood')['SalePrice'].agg([
    ('AvgPricePerNeighborhood', 'mean'),
    ('MedianPricePerNeighborhood', 'median'),
    ('StdPricePerNeighborhood', 'std')
]).reset_index()

# Calculate average lot area per neighborhood
neighborhood_lot_stats = df_train_full.groupby('Neighborhood')['LotArea'].agg([
    ('AvgLotAreaPerNeighborhood', 'mean')
]).reset_index()

# Merge neighborhood statistics
neighborhood_features = pd.merge(neighborhood_stats, neighborhood_lot_stats, on='Neighborhood')

print(f"Neighborhood statistics calculated for {len(neighborhood_features)} neighborhoods")
print("\nSample neighborhood features:")
print(neighborhood_features.head())

# Map neighborhood features to train/test sets
X_train = pd.merge(X_train, neighborhood_features, on='Neighborhood', how='left')
X_test = pd.merge(X_test, neighborhood_features, on='Neighborhood', how='left')

# Handle any missing values (neighborhoods not in training set) 
for col in ['AvgPricePerNeighborhood', 'MedianPricePerNeighborhood', 'StdPricePerNeighborhood', 'AvgLotAreaPerNeighborhood']:
    if col in X_train.columns:
        X_train[col].fillna(X_train[col].mean(), inplace=True)
        X_test[col].fillna(X_train[col].mean(), inplace=True)

print(f"✓ Added neighborhood aggregate features")

# ============================================================================
# 3. TRANSPORTATION PROXIMITY FEATURES
# ============================================================================
print("\n3. CREATING TRANSPORTATION PROXIMITY FEATURES...")

def create_proximity_features(df):
    """Create binary features for transportation proximity from Condition1 and Condition2"""
    df_new = df.copy()
    
    # Define proximity conditions
    proximity_mapping = {
        'Artery': 'Near_Artery',
        'Feedr': 'Near_Feedr',
        'RRNn': 'Near_RRNn',  # Railroad North-South
        'RRAn': 'Near_RRAn',  # Railroad Adjacent North
        'RRNe': 'Near_RRNe',  # Railroad Northeast
        'RRAe': 'Near_RRAe',  # Railroad Adjacent East
        'PosN': 'Near_PosN',  # Positive feature North
        'PosA': 'Near_PosA',  # Positive feature Adjacent
    }
    
    # Create binary features
    for condition, feature_name in proximity_mapping.items():
        df_new[feature_name] = 0
        if 'Condition1' in df_new.columns:
            df_new.loc[df_new['Condition1'] == condition, feature_name] = 1
        if 'Condition2' in df_new.columns:
            df_new.loc[df_new['Condition2'] == condition, feature_name] = 1
    
    # Create aggregate transportation score
    railroad_features = ['Near_RRNn', 'Near_RRAn', 'Near_RRNe', 'Near_RRAe']
    positive_features = ['Near_PosN', 'Near_PosA']
    
    df_new['Railroad_Proximity_Score'] = df_new[railroad_features].sum(axis=1)
    df_new['Positive_Features_Score'] = df_new[positive_features].sum(axis=1)
    df_new['Transportation_Score'] = (
        df_new['Near_Artery'].astype(int) + 
        df_new['Near_Feedr'].astype(int) + 
        df_new['Railroad_Proximity_Score'] + 
        df_new['Positive_Features_Score']
    )
    
    return df_new

X_train = create_proximity_features(X_train)
X_test = create_proximity_features(X_test)

print("✓ Created transportation proximity features")

# ============================================================================
# 4. INTERACTION FEATURES
# ============================================================================
print("\n4. CREATING INTERACTION FEATURES...")

# Neighborhood price × Living area
if 'GrLivArea' in X_train.columns and 'AvgPricePerNeighborhood' in X_train.columns:
    X_train['NeighborhoodPrice_x_GrLivArea'] = X_train['AvgPricePerNeighborhood'] * X_train['GrLivArea']
    X_test['NeighborhoodPrice_x_GrLivArea'] = X_test['AvgPricePerNeighborhood'] * X_test['GrLivArea']
    print("✓ Created NeighborhoodPrice × GrLivArea interaction")

# Neighborhood price × Overall quality
if 'OverallQual' in X_train.columns and 'AvgPricePerNeighborhood' in X_train.columns:
    X_train['NeighborhoodPrice_x_OverallQual'] = X_train['AvgPricePerNeighborhood'] * X_train['OverallQual']
    X_test['NeighborhoodPrice_x_OverallQual'] = X_test['AvgPricePerNeighborhood'] * X_test['OverallQual']
    print("✓ Created NeighborhoodPrice × OverallQual interaction")

print(f"\nTotal features after engineering: {X_train.shape[1]}")

# ============================================================================
# 5. PREPROCESSING FOR XGBOOST
# ============================================================================
print("\n5. PREPROCESSING FOR XGBOOST...")

# XGBoost can handle categorical features, but we'll use one-hot encoding for consistency
# Identify categorical and numerical features
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# One-hot encode categorical features
if len(categorical_features) > 0:
    X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
    
    # Align train and test columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print(f"Final feature count: {X_train.shape[1]}")

# Handle any remaining missing values
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_train.median(), inplace=True)

# ============================================================================
# 6. TRAIN XGBOOST MODEL
# ============================================================================
print("\n6. TRAINING XGBOOST MODEL...")

# XGBoost with tuned hyperparameters
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)

# Train model
print("Training XGBoost model...")
model.fit(X_train, y_train, verbose=False)
print("✓ Model training complete")

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(
    model, X_train, y_train, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
cv_rmse = np.sqrt(-cv_scores)
print(f"CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")

# ============================================================================
# 7. EVALUATION
# ============================================================================
print("\n7. MODEL EVALUATION...")

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTraining Set Performance:")
print(f"  MSE: {train_mse:,.2f}")
print(f"  MAE: {train_mae:,.2f}")
print(f"  R²:  {train_r2:.4f}")

print("\nTest Set Performance:")
print(f"  MSE: {test_mse:,.2f}")
print(f"  MAE: {test_mae:,.2f}")
print(f"  R²:  {test_r2:.4f}")

# ============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n8. FEATURE IMPORTANCE ANALYSIS...")

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20))

# Save feature importance
feature_importance.to_csv('results/04_paper1_railestate_feature_importance.csv', index=False)
print("✓ Saved: results/04_paper1_railestate_feature_importance.csv")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================
print("\n9. CREATING VISUALIZATIONS...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted (Test Set)
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price')
axes[0, 0].set_ylabel('Predicted Price')
axes[0, 0].set_title(f'Paper 1: RailEstate Spatial\nTest R² = {test_r2:.4f}')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residual Plot
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Price')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature Importance (Top 15)
top_features = feature_importance.head(15)
axes[1, 0].barh(range(len(top_features)), top_features['Importance'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['Feature'], fontsize=8)
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 15 Feature Importances')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. Error Distribution
axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Prediction Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Error Distribution (MAE={test_mae:,.0f})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/04_paper1_railestate_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/04_paper1_railestate_predictions.png")
plt.close()

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================
print("\n10. SAVING RESULTS...")

# Save results
results = pd.DataFrame({
    'Model': ['Paper 1: RailEstate Spatial (XGBoost)'],
    'Train_MSE': [train_mse],
    'Train_MAE': [train_mae],
    'Train_R2': [train_r2],
    'Test_MSE': [test_mse],
    'Test_MAE': [test_mae],
    'Test_R2': [test_r2],
    'CV_RMSE_Mean': [cv_rmse.mean()],
    'CV_RMSE_Std': [cv_rmse.std()]
})

results.to_csv('results/04_paper1_railestate_results.csv', index=False)
print("✓ Saved: results/04_paper1_railestate_results.csv")

# Save model
with open('models/04_paper1_railestate_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Saved: models/04_paper1_railestate_model.pkl")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PAPER 1: RAILESTATE SPATIAL FEATURE ENGINEERING - COMPLETE!")
print("="*80)
print("\nKey Results:")
print(f"• Test R²: {test_r2:.4f}")
print(f"• Test MSE: {test_mse:,.2f}")
print(f"• Test MAE: {test_mae:,.2f}")
print(f"• Cross-Val RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")
print("\nEngineered Features:")
print(f"• Neighborhood aggregate features (avg price, median, std, lot area)")
print(f"• Transportation proximity features (8 binary + scores)")
print(f"• Interaction features (neighborhood × property characteristics)")
print("\nNext Steps:")
print("• Review feature importance to understand spatial feature impact")
print("• Proceed to Paper 2: Two-Stage Clustering")
print("="*80)
