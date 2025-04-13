# Machine Learning Classification Models

This repository contains implementations of various classification algorithms demonstrating model selection, hyperparameter tuning, and performance evaluation techniques.

## Projects

### 🍷 Wine Classification Analysis  
**File**: `wine_classification_analysis.ipynb`  
**Description**: Classification of wine samples into different varieties based on chemical attributes.  
**Key Features**:
- Implemented k-NN, Decision Trees, and Random Forest classifiers
- Conducted silhouette analysis for optimal clustering
- Performed PCA for dimensionality reduction
- Achieved **95% accuracy** with optimal hyperparameters
- Comprehensive data preprocessing and feature engineering
- Rigorous model selection and evaluation metrics
- Advanced hyperparameter tuning with cross-validation
- Detailed performance visualization and interpretation
- Complete confusion matrix and ROC analysis
- Built with: Python, scikit-learn, pandas, numpy

### ⚙️ Error Classification with XGBoost  
**File**: `error_classification_xgboost.ipynb`  
**Description**: Binary classification model to predict manufacturing process errors.  
**Key Features**:
- XGBoost with GridSearchCV hyperparameter tuning
- One-hot encoding for categorical variables
- Systematic optimization of model parameters
- Improved accuracy from **77% to 87%**
- Comprehensive data preprocessing and feature engineering
- Rigorous model selection and evaluation metrics
- Advanced hyperparameter tuning with cross-validation
- Detailed performance visualization and interpretation
- Complete confusion matrix and ROC analysis
- Built with: Python, XGBoost, matplotlib

## Dataset Loading
```python
from ucimlrepo import fetch_ucirepo 

# Fetch dataset 
wine = fetch_ucirepo(id=109) 
X = wine.data.features  # Features (13 chemical properties)
y = wine.data.targets   # Targets (3 wine classes)

# View dataset info
print(wine.metadata)
print(wine.variables)

## VS Code Setup

```bash
# Clone repository
git clone https://github.com/iankamar/classification-models.git
cd classification-models

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Open in VS Code
code .

# Run scripts
python wine_classification.py  # Uses wine dataset from: https://archive.ics.uci.edu/dataset/109/wine
python error_classification.py  # Uses predictive maintenance dataset from: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

```
## License
Copyright © 2024 Ian Kamar