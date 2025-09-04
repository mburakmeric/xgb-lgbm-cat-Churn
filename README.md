# Customer Churn Prediction with Ensemble Boosting Methods

A comprehensive machine learning project that compares XGBoost, LightGBM, and CatBoost algorithms for predicting customer churn on an imbalanced dataset. This project implements best practices for handling class imbalance, hyperparameter tuning, and model evaluation.

## Project Overview

Customer churn prediction is crucial for subscription-based businesses to identify customers likely to cancel their services. This project provides a complete end-to-end solution comparing three state-of-the-art gradient boosting algorithms on a telecommunications customer dataset.

### Key Features

- **Comprehensive EDA**: In-depth exploratory data analysis with visualizations
- **Data Preprocessing**: Robust data cleaning and feature engineering pipeline
- **Imbalance Handling**: Implementation of SMOTE and class weight strategies
- **Model Comparison**: Systematic evaluation of XGBoost, LightGBM, and CatBoost
- **Hyperparameter Tuning**: Grid search with cross-validation for optimal performance
- **Advanced Evaluation**: ROC curves, precision-recall analysis, and SHAP interpretability
- **Production Ready**: Structured code with best practices and reproducible results

## Dataset

The dataset contains customer information from a telecommunications company with the following characteristics:

- **Size**: 7,044 customer records
- **Features**: 20 attributes including demographics, services, and account information
- **Target**: Binary churn indicator (Yes/No)
- **Challenge**: Imbalanced dataset requiring specialized handling techniques

### Key Features Include:
- Demographics: Gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, InternetService, OnlineSecurity, etc.
- Account: Contract type, PaymentMethod, MonthlyCharges, TotalCharges
- Tenure: Customer relationship length

## Installation

### Prerequisites
- Python 3.9 or higher
- Jupyter Notebook

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd churn
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook notebooks/churn_boosting.ipynb
```

## Usage

The main analysis is contained in the Jupyter notebook `notebooks/churn_boosting.ipynb`. The notebook is structured as follows:

1. **Data Loading & Initial Exploration**
   - Dataset overview and basic statistics
   - Missing value analysis and treatment

2. **Exploratory Data Analysis**
   - Feature distributions and correlations
   - Churn rate analysis across different segments
   - Visualization of key patterns

3. **Data Preprocessing**
   - Feature encoding and scaling
   - Train-test split with stratification
   - Pipeline creation for reproducibility

4. **Model Training & Evaluation**
   - Baseline models with class weights
   - SMOTE-enhanced models
   - Hyperparameter optimization
   - Cross-validation results

5. **Model Comparison & Selection**
   - Performance metrics comparison
   - ROC curve analysis
   - Feature importance and SHAP values

6. **Results & Insights**
   - Best model selection
   - Business recommendations
   - Model interpretability

## Results

### Model Performance Summary

| Model | Strategy | CV ROC-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1-Score |
|-------|----------|------------|--------------|----------------|-------------|---------------|
| **CatBoost** | Class Weights | **0.849** | **0.843** | 0.530 | **0.755** | **0.623** |
| XGBoost | Class Weights | 0.849 | 0.841 | **0.673** | 0.508 | 0.579 |
| XGBoost | SMOTE | 0.849 | 0.841 | 0.615 | 0.581 | 0.598 |
| LightGBM | SMOTE | 0.840 | 0.838 | 0.635 | 0.530 | 0.578 |
| LightGBM | Class Weights | 0.839 | 0.835 | 0.531 | 0.731 | 0.615 |

### Key Findings

- **CatBoost with class weights** achieved the best overall performance with the highest ROC-AUC (0.843) and recall (0.755)
- **Class weight strategies** generally outperformed SMOTE for this dataset
- All models achieved strong discriminative performance (ROC-AUC > 0.83)
- Feature importance analysis revealed contract type, tenure, and monthly charges as key predictors

## Project Structure

```
churn/
├── data/
│   ├── churn.csv                    # Raw dataset
│   └── model_comparison_results.csv # Model performance results
├── notebooks/
│   ├── churn_boosting.ipynb        # Main analysis notebook
│   └── catboost_info/              # CatBoost training logs
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Technical Implementation

### Algorithms Compared
- **XGBoost**: Extreme Gradient Boosting with advanced regularization
- **LightGBM**: Microsoft's fast gradient boosting framework
- **CatBoost**: Yandex's categorical feature-optimized boosting

### Imbalance Handling Strategies
- **Class Weights**: Automatic balancing based on class frequencies
- **SMOTE**: Synthetic Minority Oversampling Technique for data augmentation

### Evaluation Metrics
- ROC-AUC (primary metric)
- Precision, Recall, F1-Score
- Confusion Matrix analysis
- Precision-Recall curves

### Model Interpretability
- Feature importance rankings
- SHAP (SHapley Additive exPlanations) values
- Partial dependence plots

## Dependencies

Core libraries used in this project:

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Boosting Libraries**: xgboost, lightgbm, catboost
- **Interpretability**: shap
- **Environment**: jupyter

See `requirements.txt` for complete version specifications.


## License

This project is open source and available under the [MIT License](LICENSE).



---

