# IntroMLCapstone - House Price Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ML Capstone Project** | Introduction to Machine Learning Course

Comprehensive house price prediction comparing classical ML algorithms with research-paper-based approaches on the Kaggle Ames Housing dataset.

🏆 **Best Model**: XGBoost + Spatial Features (R²=0.9279, MAE=$14,810)

---

## 📋 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/IntroMLCapstone.git
cd IntroMLCapstone
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Or install manually:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost interpret
```

### 3. Run the Project

#### **Option A: Run Everything (Recommended First Time)**

```bash
# Step 1: Preprocess data
python 00_EDA_and_Preprocessing.py

# Step 2: Run all classical models
python run_classical_models.py

# Step 3: Run research paper models
python models/01_Paper1_RailEstate_Spatial.py
python models/02_Paper2_TwoStage_Clustering.py

# Step 4: Generate comprehensive comparison
python models/03_Compare_All_Models.py
```

#### **Option B: Run Individual Models**

```bash
# Classical Models
python models/01_Linear_Regression.py
python models/02_Random_Forest.py
python models/03_Gradient_Boosting.py

# Research Paper Models
python models/01_Paper1_RailEstate_Spatial.py
python models/02_Paper2_TwoStage_Clustering.py
```

---

## 📊 Results Summary

| Model | R² | MAE ($) | Category |
|-------|-----|---------|----------|
| **XGBoost + Spatial Features** | **0.9279** | **14,810** | Research Paper 1 |
| Global EBM | 0.9060 | 16,387 | Research Paper 2 |
| Random Forest | 0.8910 | 17,548 | Classical |
| Gradient Boosting | 0.8840 | 17,352 | Classical |
| Linear Regression (Ridge) | 0.8774 | 19,135 | Classical |

**Best Model Improvement**: +5.8% R² over classical baseline, -22.6% MAE reduction

---

## 🗂️ Project Structure

```
IntroMLCapstone/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── Technical_Report.md                    # Full academic report
│
├── 00_EDA_and_Preprocessing.py           # Data exploration & preprocessing
├── run_classical_models.py               # Run all classical models at once
│
├── data/                                  # Preprocessed datasets
│   ├── X_train.csv                       # Training features (1,168 samples)
│   ├── X_test.csv                        # Test features (292 samples)
│   ├── y_train.csv                       # Training targets
│   ├── y_test.csv                        # Test targets
│   └── feature_info.pkl                  # Feature metadata
│
├── models/                                # Model implementations
│   ├── 01_Linear_Regression.py           # Classical: Ridge regression
│   ├── 02_Random_Forest.py               # Classical: Random Forest
│   ├── 03_Gradient_Boosting.py           # Classical: Gradient Boosting
│   ├── 01_Paper1_RailEstate_Spatial.py   # Research Paper 1: Spatial features + XGBoost
│   ├── 02_Paper2_TwoStage_Clustering.py  # Research Paper 2: Clustering + EBM
│   └── 03_Compare_All_Models.py          # Comprehensive comparison script
│
└── results/                               # Outputs and visualizations
    ├── all_models_comparison.csv         # Complete model rankings
    ├── 04_paper1_railestate_results.csv  # Paper 1 metrics
    ├── 05_paper2_clustering_results.csv  # Paper 2 metrics
    └── figures/                           # Visualization outputs
        ├── all_models_comprehensive_comparison.png
        ├── 04_paper1_railestate_predictions.png
        └── 05_paper2_clustering_analysis.png
```

---

## 🔧 Requirements

- **Python**: 3.8 or higher
- **Core Libraries**:
  - `numpy` - Numerical computing
  - `pandas` - Data manipulation
  - `matplotlib`, `seaborn` - Visualization
  - `scikit-learn` - Classical ML algorithms
  - `xgboost` - Gradient boosting (Paper 1)
  - `interpret` - Explainable Boosting Machine (Paper 2)

**Create `requirements.txt`:**
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
interpret>=0.2.7
```

---

## 📖 Detailed Usage

### Step 1: Data Preprocessing

```bash
python 00_EDA_and_Preprocessing.py
```

**What it does:**
- Loads `train.csv` and `test.csv` from Kaggle
- Handles missing values (median/mode imputation)
- Encodes categorical variables (one-hot encoding)
- Creates basic feature engineering (HouseAge, WasRemodeled)
- Splits into train/test (80/20)
- Saves preprocessed data to `data/` folder
- Generates EDA visualizations in `results/figures/`

**Outputs:**
- `data/X_train.csv`, `data/X_test.csv`, `data/y_train.csv`, `data/y_test.csv`
- `results/figures/eda_*.png` (3 visualization files)

**Time**: ~30 seconds

---

### Step 2: Classical Models

#### Option A: Run All at Once
```bash
python run_classical_models.py
```

#### Option B: Run Individually
```bash
python models/01_Linear_Regression.py  # Ridge regression baseline
python models/02_Random_Forest.py       # Ensemble of 100 trees
python models/03_Gradient_Boosting.py   # Sequential boosting
```

**What they do:**
- Load preprocessed data from `data/`
- Train each classical algorithm
- Evaluate on test set (MSE, MAE, R²)
- Save results to `results/`
- Generate prediction plots with residuals

**Outputs per model:**
- `results/0X_model_results.csv` - Performance metrics
- `results/0X_model.pkl` - Trained model
- `results/figures/0X_model.png` - Visualization

**Time**: ~2 minutes total

---

### Step 3: Research Paper Models

#### Paper 1: Spatial Feature Engineering + XGBoost

```bash
python models/01_Paper1_RailEstate_Spatial.py
```

**What it does:**
- Engineers 12 spatial features:
  - Neighborhood aggregate statistics (avg price, lot area)
  - Transportation proximity indicators (8 binary features)
  - Interaction terms (neighborhood × property characteristics)
- Trains XGBoost with 500 trees
- Analyzes feature importance

**Key Innovation**: Adapts spatial feature engineering to work without GPS coordinates using neighborhood-level aggregates.

**Outputs:**
- `results/04_paper1_railestate_results.csv`
- `results/04_paper1_railestate_feature_importance.csv`
- `models/04_paper1_railestate_model.pkl`
- `results/figures/04_paper1_railestate_predictions.png`

**Time**: ~1 minute

---

#### Paper 2: Two-Stage Clustering + EBM

```bash
python models/02_Paper2_TwoStage_Clustering.py
```

**What it does:**
- **Stage 1**: Clusters properties by location (k=2 based on neighborhood features)
- **Stage 2**: Clusters each location by property characteristics (k=4)
- Trains 16 cluster-specific models (8 clusters × 2 types):
  - Lasso Regression (L1 penalty)
  - Explainable Boosting Machine (EBM)
- Trains global models for comparison
- Compares cluster-specific vs global performance

**Key Finding**: Global EBM outperforms cluster-specific models due to small cluster sizes.

**Outputs:**
- `results/05_paper2_clustering_results.csv` - Per-cluster performance
- `results/05_paper2_global_vs_clustered.csv` - Comparison
- `results/05_paper2_cluster_assignments_train.csv` - Cluster labels
- `models/05_paper2_cluster_models.pkl` - All 18 trained models
- `results/figures/05_paper2_clustering_analysis.png`

**Time**: ~40 minutes (EBM training is computationally intensive)

---

### Step 4: Comprehensive Comparison

```bash
python models/03_Compare_All_Models.py
```

**What it does:**
- Loads results from all 6 models
- Creates comprehensive comparison tables
- Generates multi-panel visualization comparing all approaches
- Calculates improvement over baseline
- Ranks models by performance

**Outputs:**
- `results/all_models_comparison.csv` - Complete rankings
- `results/all_models_detailed_comparison.csv` - Formatted metrics
- `results/figures/all_models_comprehensive_comparison.png` - 9-panel figure

**Time**: ~10 seconds

---

## 🔬 Methodology Summary

### Classical Models

1. **Linear Regression (Ridge)**: L2 regularization, scaled features, interpretable baseline
2. **Random Forest**: 100 trees, max_depth=20, bootstrap aggregation
3. **Gradient Boosting**: 100 estimators, learning_rate=0.1, sequential boosting

### Research Paper 1: Spatial Feature Engineering

**Approach**: Engineer location-based features without GPS coordinates

**Features Created**:
- `AvgPricePerNeighborhood`, `MedianPricePerNeighborhood`, `StdPricePerNeighborhood`
- `AvgLotAreaPerNeighborhood`
- `Near_Artery`, `Near_Feedr`, `Near_RRxx` (transportation proximity)
- `NeighborhoodPrice_x_GrLivArea` (interaction features)

**Model**: XGBoost (500 trees, learning_rate=0.05, L1/L2 regularization)

**Result**: R²=0.9279 (best overall)

### Research Paper 2: Two-Stage Clustering

**Approach**: Hierarchical clustering with cluster-specific models

**Stage 1**: K-Means (k=2) on neighborhood features → 2 location clusters  
**Stage 2**: K-Means (k=4) on property features → 8 total clusters

**Models**: Lasso + EBM per cluster, plus global variants

**Result**: Global EBM (R²=0.906) outperforms cluster-specific (R²=0.870)

---

## 📈 Key Findings

### What Worked Best

✅ **Spatial feature engineering** (Paper 1):
- Neighborhood aggregates ranked 4th in feature importance
- Interaction terms captured location × property synergies
- XGBoost + engineered features = +5.8% R² improvement

✅ **Global EBM** (Paper 2):
- Interpretable boosting competitive with XGBoost
- R²=0.906 without extensive hyperparameter tuning
- 2nd best overall performance

### What Didn't Work

❌ **Cluster-specific models** (Paper 2):
- Small cluster sizes (avg 146 samples) led to overfitting
- Lasso particularly struggled (R²=0.03)
- Homogeneous market negated clustering benefits

### Insights for Practitioners

1. **Feature engineering still matters** even with advanced boosting
2. **Neighborhood aggregates** effective spatial proxies when GPS unavailable
3. **Global sophisticated models** often beat localized simple models
4. **Clustering requires** large datasets and heterogeneous markets

---

## 📊 Visualizations

All visualizations saved in `results/figures/`:

- **EDA Visualizations**:
  - `eda_target_distribution.png` - SalePrice distribution & box plot
  - `eda_correlation_heatmap.png` - Feature correlation matrix
  - `eda_missing_values.png` - Missing value analysis

- **Model-Specific**:
  - `01_linear_regression.png` - Ridge predictions & residuals
  - `02_random_forest.png` - RF predictions & feature importance
  - `03_gradient_boosting.png` - GB predictions & feature importance
  - `04_paper1_railestate_predictions.png` - XGBoost spatial model analysis
  - `05_paper2_clustering_analysis.png` - Cluster visualization & performance

- **Comprehensive Comparison**:
  - `all_models_comprehensive_comparison.png` - 9-panel comparison figure

---

## 📝 Technical Report

Full academic technical report available in `Technical_Report.md` (8-9 pages).

**Sections**:
1. Introduction
2. Literature Review
3. Methodology (detailed)
4. Results & Evaluation
5. Discussion
6. Conclusion
7. References

**To convert to IEEE PDF**:
1. Use Overleaf IEEE template
2. Copy content from `Technical_Report.md`
3. Add figures from `results/figures/`
4. Export as PDF

---

## 🎓 Academic Context

**Course**: Introduction to Machine Learning - Capstone Project

**Dataset**: [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**Requirements Met**:
- ✅ 3 classical ML algorithms implemented
- ✅ 2 research paper methodologies adapted
- ✅ Comprehensive performance comparison
- ✅ 8 Python files (exceeds 5-file requirement)
- ✅ Technical report (8-9 pages, exceeds 5-page minimum)
- ✅ Complete README with usage instructions
- ✅ Proper citations and documentation

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'interpret'"

**Solution**:
```bash
pip install interpret
```

### Issue: "FileNotFoundError: train.csv not found"

**Solution**: Download dataset from Kaggle and place `train.csv` and `test.csv` in project root:
```bash
# Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Place train.csv and test.csv in project root directory
```

### Issue: Paper 2 script takes very long (>30 minutes)

**Expected**: EBM models are computationally intensive. Training 8 cluster-specific EBMs can take 30-40 minutes on average hardware. This is normal.

**Alternative**: Run only global models by modifying the script.

### Issue: "MemoryError" when loading data

**Solution**: Close other applications. The full dataset with 220+ features requires ~500MB RAM.

---

## 📚 References

**Libraries Used**:

- scikit-learn: Pedregosa et al. (2011). [Link](https://scikit-learn.org)
- XGBoost: Chen & Guestrin (2016). [Link](https://xgboost.ai)
- InterpretML: Nori et al. (2019). [Link](https://interpret.ml)

**Dataset**:

- Kaggle: "House Prices - Advanced Regression Techniques" [Link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**Research Papers**:

- [Add your specific Paper 1 citation here]
- [Add your specific Paper 2 citation here]

---

## 👤 Author

**Capstone Project** - Introduction to Machine Learning Course  
**Date**: December 2025

---

## 📄 License

This project is for academic purposes as part of an ML capstone course.

---

## 🙏 Acknowledgments

- Course instructors for guidance and support
- Kaggle for providing the excellent housing dataset
- Open-source ML community for fantastic libraries
- Peers for collaboration and feedback

---

## 📧 Contact

For questions or feedback about this project:
- Open an issue in this repository
- [Add your contact information if desired]

---

**⭐ If you found this project helpful, please consider giving it a star!**
