# Machine Learning Classification Models

This repository contains implementations of various classification algorithms applied to different datasets, demonstrating model selection, hyperparameter tuning, and performance evaluation techniques.

## Projects

### Wine Classification Analysis
- **File**: `wine_classification_analysis.ipynb`
- **Description**: Classification of wine samples into different varieties based on chemical attributes.
- **Techniques**: k-Nearest Neighbors (k-NN), Decision Trees, Random Forests, Principal Component Analysis (PCA)
- **Outcomes**: Achieved 96% classification accuracy with optimal hyperparameter tuning and cross-validation.

### Error Classification with XGBoost
- **File**: `error_classification_xgboost.ipynb`
- **Description**: Binary classification model to predict errors in manufacturing processes.
- **Techniques**: Gradient Boosting, XGBoost, GridSearchCV, ROC/AUC Analysis
- **Outcomes**: Improved model accuracy from 77% to 87% through systematic hyperparameter optimization.

## Implementation Details

These projects demonstrate:
- Data preprocessing and feature engineering
- Model selection and evaluation
- Hyperparameter tuning with cross-validation
- Performance visualization and interpretation
- Confusion matrix analysis

## Technologies Used

- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- XGBoost

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/iankamar/classification-models.git

# Navigate to the directory
cd classification-models

# Create a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

```
## License
Copyright © 2024 Ian Kamar. All rights reserved.