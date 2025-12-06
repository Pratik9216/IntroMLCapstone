"""
Exploratory Data Analysis and Data Preprocessing
House Prices - Advanced Regression Techniques

This script performs:
1. Data loading and initial exploration
2. Statistical analysis
3. Visualization of key features
4. Data cleaning (missing values, outliers)
5. Feature engineering
6. Train-test split
7. Preprocessing pipeline creation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("HOUSE PRICE PREDICTION - EDA AND PREPROCESSING")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n1. LOADING DATA...")
df = pd.read_csv('train.csv')
print(f"Dataset shape: {df.shape}")
print(f"Number of features: {df.shape[1] - 1}")  # Excluding target
print(f"Number of samples: {df.shape[0]}")

# ============================================================================
# 2. INITIAL DATA EXPLORATION
# ============================================================================
print("\n2. INITIAL DATA EXPLORATION")
print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# ============================================================================
# 3. TARGET VARIABLE ANALYSIS
# ============================================================================
print("\n3. TARGET VARIABLE (SalePrice) ANALYSIS")
print(f"Mean: ${df['SalePrice'].mean():,.2f}")
print(f"Median: ${df['SalePrice'].median():,.2f}")
print(f"Std Dev: ${df['SalePrice'].std():,.2f}")
print(f"Min: ${df['SalePrice'].min():,.2f}")
print(f"Max: ${df['SalePrice'].max():,.2f}")

# Check for skewness
skewness = df['SalePrice'].skew()
print(f"Skewness: {skewness:.2f}")
if skewness > 0.5:
    print("⚠️  Target variable is right-skewed - consider log transformation")

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['SalePrice'], bins=50, edgecolor='black')
axes[0].set_xlabel('Sale Price')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Sale Price')
axes[0].axvline(df['SalePrice'].mean(), color='red', linestyle='--', label='Mean')
axes[0].axvline(df['SalePrice'].median(), color='green', linestyle='--', label='Median')
axes[0].legend()

# Box plot
axes[1].boxplot(df['SalePrice'])
axes[1].set_ylabel('Sale Price')
axes[1].set_title('Box Plot of Sale Price')

plt.tight_layout()
plt.savefig('results/figures/eda_target_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/eda_target_distribution.png")
plt.close()

# ============================================================================
# 4. MISSING VALUES ANALYSIS
# ============================================================================
print("\n4. MISSING VALUES ANALYSIS")
missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percentage': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print(f"\nColumns with missing values: {len(missing_df)}")
print("\nTop 10 columns with most missing values:")
print(missing_df.head(10))

# Visualize missing values
if len(missing_df) > 0:
    plt.figure(figsize=(12, 6))
    top_missing = missing_df.head(15)
    plt.barh(top_missing['Column'], top_missing['Missing_Percentage'])
    plt.xlabel('Missing Percentage (%)')
    plt.title('Top 15 Features with Missing Values')
    plt.tight_layout()
    plt.savefig('results/figures/eda_missing_values.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/figures/eda_missing_values.png")
    plt.close()

# ============================================================================
# 5. FEATURE TYPE IDENTIFICATION
# ============================================================================
print("\n5. FEATURE TYPE IDENTIFICATION")

# Separate numerical and categorical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove Id and target from features
if 'Id' in numerical_features:
    numerical_features.remove('Id')
if 'SalePrice' in numerical_features:
    numerical_features.remove('SalePrice')

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================
print("\n6. CORRELATION ANALYSIS")

# Calculate correlations with target
correlations = df[numerical_features + ['SalePrice']].corr()['SalePrice'].sort_values(ascending=False)
print("\nTop 10 features most correlated with SalePrice:")
print(correlations.head(11)[1:])  # Exclude SalePrice itself

# Create correlation heatmap for top features
top_features = correlations.head(11).index.tolist()  # Top 10 + SalePrice
plt.figure(figsize=(12, 10))
sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap - Top 10 Features')
plt.tight_layout()
plt.savefig('results/figures/eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/eda_correlation_heatmap.png")
plt.close()

# ============================================================================
# 7. OUTLIER DETECTION
# ============================================================================
print("\n7. OUTLIER DETECTION (using IQR method)")

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers)

print("\nOutliers in top numerical features:")
for feature in correlations.head(6).index[1:]:  # Top 5 features
    n_outliers = detect_outliers_iqr(df, feature)
    print(f"{feature}: {n_outliers} outliers ({100*n_outliers/len(df):.1f}%)")

# ============================================================================
# 8. FEATURE ENGINEERING
# ============================================================================
print("\n8. FEATURE ENGINEERING")

# Create a copy for feature engineering
df_fe = df.copy()

# Age of house
if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
    df_fe['HouseAge'] = df_fe['YrSold'] - df_fe['YearBuilt']
    print("✓ Created: HouseAge")

# Remodeled flag
if 'YearRemodAdd' in df.columns and 'YearBuilt' in df.columns:
    df_fe['WasRemodeled'] = (df_fe['YearRemodAdd'] != df_fe['YearBuilt']).astype(int)
    print("✓ Created: WasRemodeled")

# Total square footage
sqft_cols = [col for col in df.columns if 'SF' in col or 'Area' in col]
if len(sqft_cols) > 0:
    print(f"✓ Found {len(sqft_cols)} square footage related columns")

# Total bathrooms
bath_cols = [col for col in df.columns if 'Bath' in col]
if len(bath_cols) > 0:
    print(f"✓ Found {len(bath_cols)} bathroom related columns")

# ============================================================================
# 9. TRAIN-TEST SPLIT
# ============================================================================
print("\n9. TRAIN-TEST SPLIT")

# Prepare features and target
X = df_fe.drop(['SalePrice', 'Id'], axis=1, errors='ignore')
y = df_fe['SalePrice']

# Update feature lists after feature engineering
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Total features: {X.shape[1]}")
print(f"Numerical: {len(numerical_features)}")
print(f"Categorical: {len(categorical_features)}")

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# 10. PREPROCESSING PIPELINE
# ============================================================================
print("\n10. CREATING PREPROCESSING PIPELINES")

# Numerical pipeline (with scaling)
numerical_pipeline_scaled = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Numerical pipeline (without scaling - for tree-based models)
numerical_pipeline_no_scale = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combined preprocessor (with scaling)
preprocessor_scaled = ColumnTransformer([
    ('num', numerical_pipeline_scaled, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Combined preprocessor (without scaling)
preprocessor_no_scale = ColumnTransformer([
    ('num', numerical_pipeline_no_scale, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

print("✓ Created preprocessing pipelines (with and without scaling)")

# ============================================================================
# 11. SAVE PREPROCESSED DATA
# ============================================================================
print("\n11. SAVING PREPROCESSED DATA")

# Save train/test splits
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("✓ Saved: data/X_train.csv")
print("✓ Saved: data/X_test.csv")
print("✓ Saved: data/y_train.csv")
print("✓ Saved: data/y_test.csv")

# Save feature lists for models
feature_info = {
    'numerical_features': numerical_features,
    'categorical_features': categorical_features
}
import pickle
with open('data/feature_info.pkl', 'wb') as f:
    pickle.dump(feature_info, f)
print("✓ Saved: data/feature_info.pkl")

# ============================================================================
# 12. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EDA AND PREPROCESSING COMPLETE!")
print("="*80)
print("\nKey Findings:")
print(f"• Dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"• Target (SalePrice): Mean=${y.mean():,.0f}, Median=${y.median():,.0f}")
print(f"• Missing values: {len(missing_df)} columns affected")
print(f"• Feature types: {len(numerical_features)} numerical, {len(categorical_features)} categorical")
print(f"• Train/Test split: {len(X_train)}/{len(X_test)} (80/20)")
print("\nNext Steps:")
print("1. Review generated visualizations in results/figures/")
print("2. Run individual model scripts (01-05)")
print("3. Compare model performances")
print("\n" + "="*80)
