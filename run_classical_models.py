"""
Run All Classical Models
Executes all three classical ML models and creates a comparison summary
"""

import subprocess
import pandas as pd
import time

print("="*80)
print("RUNNING ALL CLASSICAL ML MODELS")
print("="*80)

models = [
    ('01_Linear_Regression.py', 'Linear Regression'),
    ('02_Random_Forest.py', 'Random Forest'),
    ('03_Gradient_Boosting.py', 'Gradient Boosting')
]

for script, name in models:
    print(f"\n{'='*80}")
    print(f"Running {name}...")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', f'models/{script}'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {name} completed in {elapsed_time:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {name}:")
        print(e.stderr)
        elapsed_time = time.time() - start_time
        print(f"Failed after {elapsed_time:.2f} seconds")

# ============================================================================
# CREATE COMPARISON SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("CREATING COMPARISON SUMMARY")
print(f"{'='*80}\n")

try:
    # Load all results
    lr_results = pd.read_csv('results/01_linear_regression_results.csv')
    rf_results = pd.read_csv('results/02_random_forest_results.csv')
    gb_results = pd.read_csv('results/03_gradient_boosting_results.csv')
    
    # Combine results
    all_results = pd.concat([lr_results, rf_results, gb_results], ignore_index=True)
    
    # Select key columns for comparison
    comparison_cols = ['Model', 'Test_MSE', 'Test_MAE', 'Test_R2']
    comparison_df = all_results[comparison_cols].copy()
    
    # Sort by Test_MSE (lower is better)
    comparison_df = comparison_df.sort_values('Test_MSE')
    
    # Save comparison
    comparison_df.to_csv('results/classical_models_comparison.csv', index=False)
    
    print("Classical Models Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    print(f"\n✓ Saved: results/classical_models_comparison.csv")
    
    # Identify best model
    best_model = comparison_df.iloc[0]['Model']
    best_mse = comparison_df.iloc[0]['Test_MSE']
    best_mae = comparison_df.iloc[0]['Test_MAE']
    best_r2 = comparison_df.iloc[0]['Test_R2']
    
    print(f"\n{'='*80}")
    print("BEST CLASSICAL MODEL")
    print(f"{'='*80}")
    print(f"Model: {best_model}")
    print(f"Test MSE: {best_mse:,.2f}")
    print(f"Test MAE: {best_mae:,.2f}")
    print(f"Test R²: {best_r2:.4f}")
    print(f"{'='*80}\n")
    
except Exception as e:
    print(f"Error creating comparison: {e}")

print("\n" + "="*80)
print("ALL CLASSICAL MODELS COMPLETED")
print("="*80)
print("\nNext Steps:")
print("1. Review results in results/ directory")
print("2. Check visualizations in results/figures/")
print("3. Proceed to literature-based models (Paper 1 & 2)")
print("="*80)
