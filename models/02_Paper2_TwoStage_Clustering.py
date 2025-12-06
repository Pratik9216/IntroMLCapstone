"""
Paper 2: Two-Stage Clustering Approach
Adapted methodology from two-stage clustering paper for Kaggle House Prices dataset

This script implements:
1. Stage 1: Location-based clustering using neighborhood features (k=2)
2. Stage 2: Property-based clustering within each location cluster (k=4 each)
3. Cluster-specific models: Lasso Regression and EBM for each of 8 clusters
4. Comparison of cluster-specific vs global models

Citation: [Add Two-Stage Clustering paper citation here]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from interpret.glassbox import ExplainableBoostingRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PAPER 2: TWO-STAGE CLUSTERING APPROACH")
print("="*80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n1. LOADING PREPROCESSED DATA...")

# Load original training data for neighborhood statistics
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
# 2. CREATE NEIGHBORHOOD FEATURES FOR STAGE 1 CLUSTERING
# ============================================================================
print("\n2. CREATING NEIGHBORHOOD FEATURES FOR STAGE 1...")

# Calculate neighborhood statistics from training data only
neighborhood_stats = df_train_full.groupby('Neighborhood').agg({
    'SalePrice': 'mean',
    'LotArea': 'mean'
}).reset_index()
neighborhood_stats.columns = ['Neighborhood', 'AvgPricePerNeighborhood', 'AvgLotAreaPerNeighborhood']

print(f"Neighborhood statistics calculated for {len(neighborhood_stats)} neighborhoods")

# Map to train/test sets
X_train = pd.merge(X_train, neighborhood_stats, on='Neighborhood', how='left')
X_test = pd.merge(X_test, neighborhood_stats, on='Neighborhood', how='left')

# Handle missing values
for col in ['AvgPricePerNeighborhood', 'AvgLotAreaPerNeighborhood']:
    X_train[col].fillna(X_train[col].mean(), inplace=True)
    X_test[col].fillna(X_train[col].mean(), inplace=True)

print("✓ Neighborhood features created")

# ============================================================================
# 3. STAGE 1: LOCATION-BASED CLUSTERING (k=2)
# ============================================================================
print("\n3. STAGE 1: LOCATION-BASED CLUSTERING (k=2)...")

# Features for location clustering
location_features = ['AvgPricePerNeighborhood', 'AvgLotAreaPerNeighborhood']

# Standardize location features
scaler_location = StandardScaler()
X_train_location = scaler_location.fit_transform(X_train[location_features])
X_test_location = scaler_location.transform(X_test[location_features])

# Apply K-Means clustering (k=2)
kmeans_location = KMeans(n_clusters=2, random_state=42, n_init=10)
X_train['LocationCluster'] = kmeans_location.fit_predict(X_train_location)
X_test['LocationCluster'] = kmeans_location.predict(X_test_location)

print(f"Location Cluster Distribution (Train):")
print(X_train['LocationCluster'].value_counts().sort_index())
print(f"\nLocation Cluster Distribution (Test):")
print(X_test['LocationCluster'].value_counts().sort_index())

# ============================================================================
# 4. STAGE 2: PROPERTY-BASED CLUSTERING (k=4 per location)
# ============================================================================
print("\n4. STAGE 2: PROPERTY-BASED CLUSTERING (k=4 per location)...")

# Select property features for clustering
property_features = [
    'GrLivArea', 'TotalBsmtSF', 'OverallQual', 'OverallCond',
    'YearBuilt', 'GarageCars', 'GarageArea',
    '1stFlrSF', 'FullBath', 'BedroomAbvGr'
]

# Verify all features exist
property_features = [f for f in property_features if f in X_train.columns]
print(f"Using {len(property_features)} property features for clustering")

# Initialize cluster column
X_train['PropertyCluster'] = -1
X_test['PropertyCluster'] = -1
X_train['FinalCluster'] = -1
X_test['FinalCluster'] = -1

# Apply K-Means clustering within each location cluster
scaler_property = StandardScaler()
cluster_id = 0

for loc_cluster in [0, 1]:
    # Get data for this location cluster
    train_mask = X_train['LocationCluster'] == loc_cluster
    test_mask = X_test['LocationCluster'] == loc_cluster
    
    X_train_loc = X_train[train_mask]
    X_test_loc = X_test[test_mask]
    
    print(f"\nLocation Cluster {loc_cluster}: {train_mask.sum()} train samples, {test_mask.sum()} test samples")
    
    # Standardize property features
    X_train_prop = scaler_property.fit_transform(X_train_loc[property_features].fillna(0))
    X_test_prop = scaler_property.transform(X_test_loc[property_features].fillna(0))
    
    # Apply K-Means (k=4) for property clustering
    kmeans_property = KMeans(n_clusters=4, random_state=42, n_init=10)
    prop_clusters_train = kmeans_property.fit_predict(X_train_prop)
    prop_clusters_test = kmeans_property.predict(X_test_prop)
    
    # Assign property clusters
    X_train.loc[train_mask, 'PropertyCluster'] = prop_clusters_train
    X_test.loc[test_mask, 'PropertyCluster'] = prop_clusters_test
    
    # Create final cluster ID (location * 4 + property)
    for prop_cluster in range(4):
        X_train.loc[train_mask & (X_train['PropertyCluster'] == prop_cluster), 'FinalCluster'] = cluster_id
        X_test.loc[test_mask & (X_test['PropertyCluster'] == prop_cluster), 'FinalCluster'] = cluster_id
        
        train_count = ((X_train['LocationCluster'] == loc_cluster) & (X_train['PropertyCluster'] == prop_cluster)).sum()
        print(f"  Final Cluster {cluster_id} (Loc{loc_cluster}-Prop{prop_cluster}): {train_count} samples")
        
        cluster_id += 1

print(f"\n✓ Created {cluster_id} final clusters (2 location × 4 property)")

# ============================================================================
# 5. PREPARE DATA FOR MODELING
# ============================================================================
print("\n5. PREPARING DATA FOR MODELING...")

# Keep original data before one-hot encoding for cluster assignments
cluster_assignments_train = X_train[['LocationCluster', 'PropertyCluster', 'FinalCluster']].copy()
cluster_assignments_test = X_test[['LocationCluster', 'PropertyCluster', 'FinalCluster']].copy()

# Remove cluster columns before preprocessing
X_train_model = X_train.drop(['LocationCluster', 'PropertyCluster', 'FinalCluster'], axis=1, errors='ignore')
X_test_model = X_test.drop(['LocationCluster', 'PropertyCluster', 'FinalCluster'], axis=1, errors='ignore')

# One-hot encode categorical features
categorical_features = X_train_model.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train_model.select_dtypes(include=[np.number]).columns.tolist()

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

if len(categorical_features) > 0:
    X_train_model = pd.get_dummies(X_train_model, columns=categorical_features, drop_first=True)
    X_test_model = pd.get_dummies(X_test_model, columns=categorical_features, drop_first=True)
    
    # Align columns
    X_train_model, X_test_model = X_train_model.align(X_test_model, join='left', axis=1, fill_value=0)

# Handle missing values
X_train_model.fillna(X_train_model.median(), inplace=True)
X_test_model.fillna(X_train_model.median(), inplace=True)

print(f"Final feature count: {X_train_model.shape[1]}")

# ============================================================================
# 6. TRAIN CLUSTER-SPECIFIC MODELS
# ============================================================================
print("\n6. TRAINING CLUSTER-SPECIFIC MODELS...")

cluster_models = {}
cluster_results = []

for cluster_id in range(8):
    print(f"\n--- Cluster {cluster_id} ---")
    
    # Get data for this cluster
    train_mask = X_train['FinalCluster'] == cluster_id
    test_mask = X_test['FinalCluster'] == cluster_id
    
    X_train_cluster = X_train_model[train_mask]
    y_train_cluster = y_train[train_mask]
    X_test_cluster = X_test_model[test_mask]
    y_test_cluster = y_test[test_mask]
    
    print(f"Train samples: {len(X_train_cluster)}, Test samples: {len(X_test_cluster)}")
    
    if len(X_train_cluster) < 10 or len(X_test_cluster) < 3:
        print(f"⚠️ Too few samples, skipping cluster {cluster_id}")
        continue
    
    # Model 1: Lasso Regression (L1 penalty)
    lasso_model = Lasso(alpha=100, random_state=42, max_iter=10000)
    lasso_model.fit(X_train_cluster, y_train_cluster)
    
    lasso_train_pred = lasso_model.predict(X_train_cluster)
    lasso_test_pred = lasso_model.predict(X_test_cluster)
    
    lasso_train_r2 = r2_score(y_train_cluster, lasso_train_pred)
    lasso_test_r2 = r2_score(y_test_cluster, lasso_test_pred)
    lasso_test_mse = mean_squared_error(y_test_cluster, lasso_test_pred)
    lasso_test_mae = mean_absolute_error(y_test_cluster, lasso_test_pred)
    
    print(f"Lasso - Train R²: {lasso_train_r2:.4f}, Test R²: {lasso_test_r2:.4f}, MAE: {lasso_test_mae:.2f}")
    
    # Model 2: Explainable Boosting Machine (EBM)
    ebm_model = ExplainableBoostingRegressor(random_state=42, n_jobs=-1)
    ebm_model.fit(X_train_cluster, y_train_cluster)
    
    ebm_train_pred = ebm_model.predict(X_train_cluster)
    ebm_test_pred = ebm_model.predict(X_test_cluster)
    
    ebm_train_r2 = r2_score(y_train_cluster, ebm_train_pred)
    ebm_test_r2 = r2_score(y_test_cluster, ebm_test_pred)
    ebm_test_mse = mean_squared_error(y_test_cluster, ebm_test_pred)
    ebm_test_mae = mean_absolute_error(y_test_cluster, ebm_test_pred)
    
    print(f"EBM    - Train R²: {ebm_train_r2:.4f}, Test R²: {ebm_test_r2:.4f}, MAE: {ebm_test_mae:.2f}")
    
    # Save models
    cluster_models[f'cluster_{cluster_id}_lasso'] = lasso_model
    cluster_models[f'cluster_{cluster_id}_ebm'] = ebm_model
    
    # Save results
    cluster_results.append({
        'Cluster': cluster_id,
        'Model': 'Lasso',
        'Train_Samples': len(X_train_cluster),
        'Test_Samples': len(X_test_cluster),
        'Train_R2': lasso_train_r2,
        'Test_R2': lasso_test_r2,
        'Test_MSE': lasso_test_mse,
        'Test_MAE': lasso_test_mae
    })
    
    cluster_results.append({
        'Cluster': cluster_id,
        'Model': 'EBM',
        'Train_Samples': len(X_train_cluster),
        'Test_Samples': len(X_test_cluster),
        'Train_R2': ebm_train_r2,
        'Test_R2': ebm_test_r2,
        'Test_MSE': ebm_test_mse,
        'Test_MAE': ebm_test_mae
    })

print(f"\n✓ Trained {len(cluster_models)} cluster-specific models")

# ============================================================================
# 7. GLOBAL MODELS FOR COMPARISON
# ============================================================================
print("\n7. TRAINING GLOBAL MODELS (for comparison)...")

# Global Lasso
global_lasso = Lasso(alpha=100, random_state=42, max_iter=10000)
global_lasso.fit(X_train_model, y_train)

global_lasso_test_pred = global_lasso.predict(X_test_model)
global_lasso_test_r2 = r2_score(y_test, global_lasso_test_pred)
global_lasso_test_mse = mean_squared_error(y_test, global_lasso_test_pred)
global_lasso_test_mae = mean_absolute_error(y_test, global_lasso_test_pred)

print(f"Global Lasso - Test R²: {global_lasso_test_r2:.4f}, MSE: {global_lasso_test_mse:,.2f}, MAE: {global_lasso_test_mae:.2f}")

# Global EBM
global_ebm = ExplainableBoostingRegressor(random_state=42, n_jobs=-1)
global_ebm.fit(X_train_model, y_train)

global_ebm_test_pred = global_ebm.predict(X_test_model)
global_ebm_test_r2 = r2_score(y_test, global_ebm_test_pred)
global_ebm_test_mse = mean_squared_error(y_test, global_ebm_test_pred)
global_ebm_test_mae = mean_absolute_error(y_test, global_ebm_test_pred)

print(f"Global EBM    - Test R²: {global_ebm_test_r2:.4f}, MSE: {global_ebm_test_mse:,.2f}, MAE: {global_ebm_test_mae:.2f}")

# Save global models
cluster_models['global_lasso'] = global_lasso
cluster_models['global_ebm'] = global_ebm

# ============================================================================
# 8. AGGREGATE CLUSTER PREDICTIONS
# ============================================================================
print("\n8. AGGREGATING CLUSTER-SPECIFIC PREDICTIONS...")

# Create prediction arrays
cluster_lasso_predictions_test = np.zeros(len(y_test))
cluster_ebm_predictions_test = np.zeros(len(y_test))

for cluster_id in range(8):
    test_mask = X_test['FinalCluster'] == cluster_id
    
    if test_mask.sum() > 0:
        X_test_cluster = X_test_model[test_mask]
        
        # Get predictions from cluster-specific models
        if f'cluster_{cluster_id}_lasso' in cluster_models:
            cluster_lasso_predictions_test[test_mask] = cluster_models[f'cluster_{cluster_id}_lasso'].predict(X_test_cluster)
        
        if f'cluster_{cluster_id}_ebm' in cluster_models:
            cluster_ebm_predictions_test[test_mask] = cluster_models[f'cluster_{cluster_id}_ebm'].predict(X_test_cluster)

# Calculate overall cluster-specific model performance
cluster_lasso_overall_r2 = r2_score(y_test, cluster_lasso_predictions_test)
cluster_lasso_overall_mse = mean_squared_error(y_test, cluster_lasso_predictions_test)
cluster_lasso_overall_mae = mean_absolute_error(y_test, cluster_lasso_predictions_test)

cluster_ebm_overall_r2 = r2_score(y_test, cluster_ebm_predictions_test)
cluster_ebm_overall_mse = mean_squared_error(y_test, cluster_ebm_predictions_test)
cluster_ebm_overall_mae = mean_absolute_error(y_test, cluster_ebm_predictions_test)

print(f"\nCluster-Specific Lasso (Overall) - R²: {cluster_lasso_overall_r2:.4f}, MSE: {cluster_lasso_overall_mse:,.2f}, MAE: {cluster_lasso_overall_mae:.2f}")
print(f"Cluster-Specific EBM (Overall)   - R²: {cluster_ebm_overall_r2:.4f}, MSE: {cluster_ebm_overall_mse:,.2f}, MAE: {cluster_ebm_overall_mae:.2f}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n9. SAVING RESULTS...")

# Cluster-specific results
cluster_results_df = pd.DataFrame(cluster_results)
cluster_results_df.to_csv('results/05_paper2_clustering_results.csv', index=False)
print("✓ Saved: results/05_paper2_clustering_results.csv")

# Cluster assignments
cluster_assignments_train['y_train'] = y_train
cluster_assignments_test['y_test'] = y_test
cluster_assignments_train.to_csv('results/05_paper2_cluster_assignments_train.csv', index=False)
cluster_assignments_test.to_csv('results/05_paper2_cluster_assignments_test.csv', index=False)
print("✓ Saved: results/05_paper2_cluster_assignments_train.csv")
print("✓ Saved: results/05_paper2_cluster_assignments_test.csv")

# Global vs Clustered comparison
comparison_df = pd.DataFrame({
    'Model': ['Global Lasso', 'Global EBM', 'Cluster-Specific Lasso (Aggregated)', 'Cluster-Specific EBM (Aggregated)'],
    'Test_R2': [global_lasso_test_r2, global_ebm_test_r2, cluster_lasso_overall_r2, cluster_ebm_overall_r2],
    'Test_MSE': [global_lasso_test_mse, global_ebm_test_mse, cluster_lasso_overall_mse, cluster_ebm_overall_mse],
    'Test_MAE': [global_lasso_test_mae, global_ebm_test_mae, cluster_lasso_overall_mae, cluster_ebm_overall_mae]
})
comparison_df.to_csv('results/05_paper2_global_vs_clustered.csv', index=False)
print("✓ Saved: results/05_paper2_global_vs_clustered.csv")

# Save all models
with open('models/05_paper2_cluster_models.pkl', 'wb') as f:
    pickle.dump(cluster_models, f)
print("✓ Saved: models/05_paper2_cluster_models.pkl")

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n10. CREATING VISUALIZATIONS...")

fig = plt.figure(figsize=(16, 12))

# 1. Cluster distribution
ax1 = plt.subplot(3, 3, 1)
cluster_counts = X_train['FinalCluster'].value_counts().sort_index()
ax1.bar(cluster_counts.index, cluster_counts.values)
ax1.set_xlabel('Cluster ID')
ax1.set_ylabel('Number of Samples')
ax1.set_title('Cluster Distribution (Training Set)')
ax1.grid(True, alpha=0.3)

# 2. Average price per cluster
ax2 = plt.subplot(3, 3, 2)
cluster_avg_prices = []
for i in range(8):
    mask = X_train['FinalCluster'] == i
    if mask.sum() > 0:
        cluster_avg_prices.append(y_train[mask].mean())
    else:
        cluster_avg_prices.append(0)
ax2.bar(range(8), cluster_avg_prices)
ax2.set_xlabel('Cluster ID')
ax2.set_ylabel('Average Price')
ax2.set_title('Average Sale Price by Cluster')
ax2.grid(True, alpha=0.3)

# 3. Cluster performance comparison (R²)
ax3 = plt.subplot(3, 3, 3)
cluster_r2_by_id = cluster_results_df.groupby('Cluster')['Test_R2'].max()
colors = ['red' if r2 < global_ebm_test_r2 else 'green' for r2 in cluster_r2_by_id]
ax3.bar(cluster_r2_by_id.index, cluster_r2_by_id.values, color=colors, alpha=0.7)
ax3.axhline(y=global_ebm_test_r2, color='blue', linestyle='--', label='Global EBM')
ax3.set_xlabel('Cluster ID')
ax3.set_ylabel('Test R²')
ax3.set_title('Best R² per Cluster vs Global')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Global vs Clustered - Bar chart
ax4 = plt.subplot(3, 3, 4)
comparison_models = comparison_df['Model'].str.replace(' (Aggregated)', '').str.replace('Cluster-Specific ', 'Clust-')
x_pos = np.arange(len(comparison_models))
ax4.bar(x_pos, comparison_df['Test_R2'])
ax4.set_xticks(x_pos)
ax4.set_xticklabels(comparison_models, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Test R²')
ax4.set_title('Global vs Cluster-Specific Models')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Cluster-specific EBM predictions
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(y_test, cluster_ebm_predictions_test, alpha=0.5, edgecolors='k', linewidth=0.5)
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax5.set_xlabel('Actual Price')
ax5.set_ylabel('Predicted Price')
ax5.set_title(f'Cluster-Specific EBM\nR² = {cluster_ebm_overall_r2:.4f}')
ax5.grid(True, alpha=0.3)

# 6. Global EBM predictions
ax6 = plt.subplot(3, 3, 6)
ax6.scatter(y_test, global_ebm_test_pred, alpha=0.5, edgecolors='k', linewidth=0.5, color='orange')
ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax6.set_xlabel('Actual Price')
ax6.set_ylabel('Predicted Price')
ax6.set_title(f'Global EBM\nR² = {global_ebm_test_r2:.4f}')
ax6.grid(True, alpha=0.3)

# 7. Residuals - Cluster-specific EBM
ax7 = plt.subplot(3, 3, 7)
residuals_cluster = y_test - cluster_ebm_predictions_test
ax7.scatter(cluster_ebm_predictions_test, residuals_cluster, alpha=0.5, edgecolors='k', linewidth=0.5)
ax7.axhline(y=0, color='r', linestyle='--', lw=2)
ax7.set_xlabel('Predicted Price')
ax7.set_ylabel('Residuals')
ax7.set_title('Residuals: Cluster-Specific EBM')
ax7.grid(True, alpha=0.3)

# 8. Location clusters in feature space
ax8 = plt.subplot(3, 3, 8)
for loc_cluster in [0, 1]:
    mask = X_train['LocationCluster'] == loc_cluster
    ax8.scatter(X_train.loc[mask, 'AvgPricePerNeighborhood'], 
               X_train.loc[mask, 'AvgLotAreaPerNeighborhood'], 
               label=f'Location {loc_cluster}', alpha=0.6)
ax8.set_xlabel('Avg Price per Neighborhood')
ax8.set_ylabel('Avg Lot Area per Neighborhood')
ax8.set_title('Stage 1: Location Clusters')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Model comparison table (text)
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
table_data = []
for _, row in comparison_df.iterrows():
    table_data.append([
        row['Model'][:20],
        f"{row['Test_R2']:.3f}",
        f"{int(row['Test_MAE']):,}"
    ])
table = ax9.table(cellText=table_data, colLabels=['Model', 'R²', 'MAE'], 
                  loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)
ax9.set_title('Performance Summary', pad=20)

plt.tight_layout()
plt.savefig('results/figures/05_paper2_clustering_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/05_paper2_clustering_analysis.png")
plt.close()

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PAPER 2: TWO-STAGE CLUSTERING APPROACH - COMPLETE!")
print("="*80)
print("\nClustering Summary:")
print(f"• Stage 1: 2 location-based clusters")
print(f"• Stage 2: 4 property-based clusters per location (8 total)")
print(f"• Models trained: {len(cluster_models)} (16 cluster-specific + 2 global)")
print("\nGlobal Models Performance:")
print(f"• Global Lasso - R²: {global_lasso_test_r2:.4f}, MAE: {global_lasso_test_mae:.2f}")
print(f"• Global EBM   - R²: {global_ebm_test_r2:.4f}, MAE: {global_ebm_test_mae:.2f}")
print("\nCluster-Specific Models Performance (Aggregated):")
print(f"• Cluster Lasso - R²: {cluster_lasso_overall_r2:.4f}, MAE: {cluster_lasso_overall_mae:.2f}")
print(f"• Cluster EBM   - R²: {cluster_ebm_overall_r2:.4f}, MAE: {cluster_ebm_overall_mae:.2f}")

# Determine if clustering helped
if cluster_ebm_overall_r2 > global_ebm_test_r2:
    improvement = (cluster_ebm_overall_r2 - global_ebm_test_r2) * 100
    print(f"\n✓ Cluster-specific approach improved R² by {improvement:.2f}%")
else:
    decline = (global_ebm_test_r2 - cluster_ebm_overall_r2) * 100
    print(f"\n⚠️ Global model performed {decline:.2f}% better than cluster-specific")

print("\nNext Steps:")
print("• Review cluster assignments and characteristics")
print("• Proceed to comprehensive model comparison")
print("="*80)
