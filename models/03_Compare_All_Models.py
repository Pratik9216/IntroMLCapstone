"""
Comprehensive Model Comparison
Compares all implemented models: Classical (3) + Research Papers (2+)

This script:
1. Loads results from all models
2. Creates comprehensive comparison tables
3. Generates visualizations comparing all models
4. Identifies best performing models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# ============================================================================
# 1. LOAD ALL MODEL RESULTS
# ============================================================================
print("\n1. LOADING ALL MODEL RESULTS...")

# Classical models
lr_results = pd.read_csv('results/01_linear_regression_results.csv')
rf_results = pd.read_csv('results/02_random_forest_results.csv')
gb_results = pd.read_csv('results/03_gradient_boosting_results.csv')

# Paper 1: RailEstateX
paper1_results = pd.read_csv('results/04_paper1_railestate_results.csv')

# Paper 2: Global models (best performers)
paper2_global = pd.read_csv('results/05_paper2_global_vs_clustered.csv')

print("✓ Loaded results from all models")

# ============================================================================
# 2. CREATE COMPREHENSIVE COMPARISON TABLE
# ============================================================================
print("\n2. CREATING COMPREHENSIVE COMPARISON TABLE...")

# Combine all results
all_models = []

# Add classical models
for df in [lr_results, rf_results, gb_results]:
    all_models.append({
        'Model': df['Model'].values[0],
        'Test_MSE': df['Test_MSE'].values[0],
        'Test_MAE': df['Test_MAE'].values[0],
        'Test_R2': df['Test_R2'].values[0],
        'Category': 'Classical'
    })

# Add Paper 1 model
all_models.append({
    'Model': 'Paper 1: RailEstate Spatial (XGBoost)',
    'Test_MSE': paper1_results['Test_MSE'].values[0],
    'Test_MAE': paper1_results['Test_MAE'].values[0],
    'Test_R2': paper1_results['Test_R2'].values[0],
    'Category': 'Research Paper 1'
})

# Add Paper 2 best models
all_models.append({
    'Model': 'Paper 2: Global EBM',
    'Test_MSE': paper2_global[paper2_global['Model'] == 'Global EBM']['Test_MSE'].values[0],
    'Test_MAE': paper2_global[paper2_global['Model'] == 'Global EBM']['Test_MAE'].values[0],
    'Test_R2': paper2_global[paper2_global['Model'] == 'Global EBM']['Test_R2'].values[0],
    'Category': 'Research Paper 2'
})

all_models.append({
    'Model': 'Paper 2: Cluster-Specific EBM',
    'Test_MSE': paper2_global[paper2_global['Model'] == 'Cluster-Specific EBM (Aggregated)']['Test_MSE'].values[0],
    'Test_MAE': paper2_global[paper2_global['Model'] == 'Cluster-Specific EBM (Aggregated)']['Test_MAE'].values[0],
    'Test_R2': paper2_global[paper2_global['Model'] == 'Cluster-Specific EBM (Aggregated)']['Test_R2'].values[0],
    'Category': 'Research Paper 2'
})

# Create DataFrame
comparison_df = pd.DataFrame(all_models)

# Sort by R² (descending)
comparison_df = comparison_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)

# Add  rank
comparison_df['Rank'] = range(1, len(comparison_df) + 1)

# Calculate RMSE
comparison_df['Test_RMSE'] = np.sqrt(comparison_df['Test_MSE'])

print("\nModel Comparison (sorted by R²):")
print(comparison_df[['Rank', 'Model', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'Category']].to_string(index=False))

# Save comparison
comparison_df.to_csv('results/all_models_comparison.csv', index=False)
print(f"\n✓ Saved: results/all_models_comparison.csv")

# ============================================================================
# 3. PERFORMANCE ANALYSIS
# ============================================================================
print("\n3. PERFORMANCE ANALYSIS...")

best_model = comparison_df.iloc[0]
print(f"\n{'='*80}")
print(f"BEST PERFORMING MODEL")
print(f"{'='*80}")
print(f"Model: {best_model['Model']}")
print(f"Category: {best_model['Category']}")
print(f"Test R²: {best_model['Test_R2']:.4f}")
print(f"Test RMSE: ${best_model['Test_RMSE']:,.2f}")
print(f"Test MAE: ${best_model['Test_MAE']:,.2f}")
print(f"{'='*80}")

# Calculate improvements over baseline (Linear Regression)
baseline = comparison_df[comparison_df['Model'].str.contains('Linear Regression')].iloc[0]
print(f"\nImprovement over Baseline (Linear Regression):")
for idx, row in comparison_df.iterrows():
    if 'Linear Regression' not in row['Model']:
        r2_improvement = ((row['Test_R2'] - baseline['Test_R2']) / baseline['Test_R2']) * 100
        mae_improvement = ((baseline['Test_MAE'] - row['Test_MAE']) / baseline['Test_MAE']) * 100
        print(f"  {row['Model'][:40]:40s}: R² +{r2_improvement:5.2f}%  |  MAE {mae_improvement:+6.2f}%")

# ============================================================================
# 4. CREATE COMPREHENSIVE VISUALIZATIONS
# ============================================================================
print("\n4. CREATING COMPREHENSIVE VISUALIZATIONS...")

fig = plt.figure(figsize=(20, 12))

# 1. R² Comparison
ax1 = plt.subplot(3, 3, 1)
colors = ['#2E7D32' if cat == 'Classical' else '#1976D2' if cat == 'Research Paper 1' else '#D32F2F' 
          for cat in comparison_df['Category']]
bars = ax1.barh(range(len(comparison_df)), comparison_df['Test_R2'], color=colors, alpha=0.8)
ax1.set_yticks(range(len(comparison_df)))
ax1.set_yticklabels([m[:30] for m in comparison_df['Model']], fontsize=9)
ax1.set_xlabel('Test R²')
ax1.set_title('Model Comparison: R² Score')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')
ax1.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='R²=0.9')
ax1.legend(fontsize=8)

# 2. MAE Comparison
ax2 = plt.subplot(3, 3, 2)
bars = ax2.barh(range(len(comparison_df)), comparison_df['Test_MAE'], color=colors, alpha=0.8)
ax2.set_yticks(range(len(comparison_df)))
ax2.set_yticklabels([m[:30] for m in comparison_df['Model']], fontsize=9)
ax2.set_xlabel('Test MAE ($)')
ax2.set_title('Model Comparison: Mean Absolute Error')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# 3. RMSE Comparison
ax3 = plt.subplot(3, 3, 3)
bars = ax3.barh(range(len(comparison_df)), comparison_df['Test_RMSE'], color=colors, alpha=0.8)
ax3.set_yticks(range(len(comparison_df)))
ax3.set_yticklabels([m[:30] for m in comparison_df['Model']], fontsize=9)
ax3.set_xlabel('Test RMSE ($)')
ax3.set_title('Model Comparison: Root Mean Squared Error')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# 4. Performance by Category
ax4 = plt.subplot(3, 3, 4)
category_performance = comparison_df.groupby('Category')['Test_R2'].mean().sort_values(ascending=False)
colors_cat = ['#2E7D32', '#1976D2', '#D32F2F']
ax4.bar(range(len(category_performance)), category_performance.values, color=colors_cat, alpha=0.8)
ax4.set_xticks(range(len(category_performance)))
ax4.set_xticklabels(category_performance.index, rotation=15, ha='right', fontsize=9)
ax4.set_ylabel('Average Test R²')
ax4.set_title('Average Performance by Category')
ax4.grid(True, alpha=0.3, axis='y')

# 5. R² vs MAE Scatter
ax5 = plt.subplot(3, 3, 5)
for cat, color in zip(['Classical', 'Research Paper 1', 'Research Paper 2'], 
                       ['#2E7D32', '#1976D2', '#D32F2F']):
    mask = comparison_df['Category'] == cat
    ax5.scatter(comparison_df[mask]['Test_R2'], comparison_df[mask]['Test_MAE'], 
               label=cat, s=150, alpha=0.7, color=color, edgecolors='black', linewidths=1)
ax5.set_xlabel('Test R²')
ax5.set_ylabel('Test MAE ($)')
ax5.set_title('R² vs MAE Trade-off')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# 6. Model Ranking
ax6 = plt.subplot(3, 3, 6)
ax6.axis('off')
ranking_text = "MODEL RANKING\n" + "="*40 + "\n\n"
for idx, row in comparison_df.iterrows():
    ranking_text += f"{row['Rank']}. {row['Model'][:35]}\n"
    ranking_text += f"   R²: {row['Test_R2']:.4f} | MAE: ${row['Test_MAE']:,.0f}\n\n"
ax6.text(0.1, 0.95, ranking_text, fontsize=9, family='monospace', 
         verticalalignment='top', transform=ax6.transAxes)
ax6.set_title('Complete Model Ranking', pad=10, fontsize=12, fontweight='bold')

# 7. Improvement over Baseline
ax7 = plt.subplot(3, 3, 7)
improvements = []
model_names = []
for idx, row in comparison_df.iterrows():
    if 'Linear Regression' not in row['Model']:
        improvement = ((row['Test_R2'] - baseline['Test_R2']) / baseline['Test_R2']) * 100
        improvements.append(improvement)
        model_names.append(row['Model'][:25])
        
ax7.barh(range(len(improvements)), improvements, color='teal', alpha=0.7)
ax7.set_yticks(range(len(improvements)))
ax7.set_yticklabels(model_names, fontsize=9)
ax7.set_xlabel('R² Improvement over Baseline (%)')
ax7.set_title('Models vs Baseline (Linear Regression)')
ax7.invert_yaxis()
ax7.grid(True, alpha=0.3, axis='x')
ax7.axvline(x=0, color='black', linestyle='-', linewidth=1)

# 8. Performance Summary Table
ax8 = plt.subplot(3, 3, 8)
ax8.axis('off')
table_data = []
for idx, row in comparison_df.head(5).iterrows():
    table_data.append([
        f"{row['Rank']}",
        row['Model'][:25],
        f"{row['Test_R2']:.3f}",
        f"{int(row['Test_MAE']):,}"
    ])
table = ax8.table(cellText=table_data, colLabels=['#', 'Model', 'R²', 'MAE'], 
                  loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2.5)
ax8.set_title('Top 5 Models Summary', pad=20, fontsize=11, fontweight='bold')

# 9. Error Distribution Comparison (text summary)
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = "KEY INSIGHTS\n" + "="*40 + "\n\n"
summary_text += f"Best Model:\n{best_model['Model'][:35]}\n\n"
summary_text += f"Best R²: {best_model['Test_R2']:.4f}\n"
summary_text += f"Best MAE: ${best_model['Test_MAE']:,.0f}\n\n"
summary_text += f"Category Rankings:\n"
for i, (cat, r2) in enumerate(category_performance.items(), 1):
    summary_text += f"{i}. {cat}: R²={r2:.4f}\n"
summary_text += f"\nTotal Models Compared: {len(comparison_df)}\n"
summary_text += f"Classical: {(comparison_df['Category']=='Classical').sum()}\n"
summary_text += f"Research Papers: {(comparison_df['Category']!='Classical').sum()}\n"

ax9.text(0.1, 0.95, summary_text, fontsize=10, family='monospace', 
         verticalalignment='top', transform=ax9.transAxes)
ax9.set_title('Performance Summary', pad=10, fontsize=12, fontweight='bold')

plt.suptitle('Comprehensive Model Performance Comparison - House Price Prediction', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('results/figures/all_models_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/all_models_comprehensive_comparison.png")
plt.close()

# ============================================================================
# 5. DETAILED COMPARISON TABLE
# ============================================================================
print("\n5. CREATING DETAILED COMPARISON TABLE...")

# Create detailed table with formatted values
detailed_comparison = comparison_df.copy()
detailed_comparison['Test_R2_fmt'] = detailed_comparison['Test_R2'].apply(lambda x: f"{x:.4f}")
detailed_comparison['Test_RMSE_fmt'] = detailed_comparison['Test_RMSE'].apply(lambda x: f"${x:,.2f}")
detailed_comparison['Test_MAE_fmt'] = detailed_comparison['Test_MAE'].apply(lambda x: f"${x:,.2f}")

# Save detailed comparison
detailed_comparison[['Rank', 'Model', 'Category', 'Test_R2_fmt', 'Test_RMSE_fmt', 'Test_MAE_fmt']].to_csv(
    'results/all_models_detailed_comparison.csv', index=False
)
print("✓ Saved: results/all_models_detailed_comparison.csv")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON - COMPLETE!")
print("="*80)
print("\nKey Findings:")
print(f"• Total models compared: {len(comparison_df)}")
print(f"• Best performing model: {best_model['Model']}")
print(f"• Best R² score: {best_model['Test_R2']:.4f}")
print(f"• Best MAE: ${best_model['Test_MAE']:,.2f}")

print("\nModel Categories:")
for cat in comparison_df['Category'].unique():
    count = (comparison_df['Category'] == cat).sum()
    avg_r2 = comparison_df[comparison_df['Category'] == cat]['Test_R2'].mean()
    print(f"• {cat}: {count} models, Average R²={avg_r2:.4f}")

print("\nOutput Files:")
print("• results/all_models_comparison.csv")
print("• results/all_models_detailed_comparison.csv")
print("• results/figures/all_models_comprehensive_comparison.png")

print("\nNext Steps:")
print("1. Review comprehensive comparison visualization")
print("2. Analyze feature importance across models")
print("3. Prepare technical report with findings")
print("4. Update README with complete results")
print("="*80)
