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
- Achieved **96% accuracy** with optimal hyperparameters

### ⚙️ Error Classification with XGBoost
**File**: `error_classification_xgboost.ipynb`  
**Description**: Binary classification model to predict manufacturing process errors.  
**Key Features**:
- XGBoost with GridSearchCV hyperparameter tuning
- One-hot encoding for categorical variables
- Systematic optimization of model parameters
- Improved accuracy from **77% to 87%**

## Dataset Loading
To use the Wine dataset:
```python
from ucimlrepo import fetch_ucirepo 

# Fetch dataset 
wine = fetch_ucirepo(id=109) 
X = wine.data.features  # Features (13 chemical properties)
y = wine.data.targets   # Targets (3 wine classes)

# View dataset info
print(wine.metadata)
print(wine.variables)

## Implementation Details

These projects demonstrate:
- Comprehensive data preprocessing and feature engineering
- Rigorous model selection and evaluation metrics
- Advanced hyperparameter tuning with cross-validation
- Detailed performance visualization and interpretation
- Complete confusion matrix and ROC analysis

## Technology Stack

<img src="https://img.icons8.com/color/24/000000/python.png" width="18"/> Python 3.7+  
<img src="https://img.icons8.com/color/24/000000/scikit-learn.png" width="18"/> scikit-learn  
<img src="https://img.icons8.com/color/24/000000/pandas.png" width="18"/> pandas  
<img src="https://img.icons8.com/color/24/000000/numpy.png" width="18"/> numpy  
<img src="https://img.icons8.com/color/24/000000/matplotlib.png" width="18"/> matplotlib  
<img src="https://img.icons8.com/color/24/000000/seaborn.png" width="18"/> seaborn  
<img src="https://img.icons8.com/color/24/000000/xgboost.png" width="18"/> XGBoost

## Setup and Installation

```bash
# Clone repository
git clone https://github.com/iankamar/classification-models.git
cd classification-models

# Set up virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook

```
## License
Copyright © 2024 Ian Kamar. All rights reserved.