# Wine Classification Model

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
python wine_classification.ipynb  # Uses wine dataset from: https://archive.ics.uci.edu/dataset/109/wine

```

### Data Licenses
- **Wine Dataset**: From UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/109/wine)  
  - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  - Citation: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.
