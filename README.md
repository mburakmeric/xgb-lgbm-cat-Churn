# Customer Churn Prediction: Boosting Algorithms Comparison

A comprehensive machine learning project comparing XGBoost, LightGBM, and CatBoost for predicting customer churn in telecommunications. Features advanced imbalanced dataset handling, hyperparameter optimization, and model interpretability analysis.

## Key Features

- **Model Comparison**: Systematic evaluation of three gradient boosting algorithms
- **Imbalance Handling**: SMOTE vs class weighting strategy comparison  
- **Comprehensive EDA**: Feature analysis and correlation studies
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Model Interpretability**: SHAP analysis for business insights
- **Production Pipeline**: End-to-end preprocessing and evaluation framework

## Dataset

Telecommunications customer dataset with 7,044 records and 20 features:
- **Target**: Binary churn indicator (26.5% churn rate)
- **Features**: Demographics, service portfolio, account information, tenure
- **Challenge**: Imbalanced classification requiring specialized techniques

## Methodology

**Two-Branch Experimental Design**
- **Branch A**: Class weight optimization for imbalance handling
- **Branch B**: SMOTE (Synthetic Minority Oversampling) implementation

**Model Evaluation**
- 5-fold stratified cross-validation
- ROC-AUC, Precision, Recall, F1-Score metrics
- SHAP analysis for feature importance and interpretability

## Results

| Model | Strategy | CV ROC-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1-Score |
|-------|----------|------------|--------------|----------------|-------------|---------------|
| **CatBoost** | **Class Weights** | **0.849** | **0.843** | 0.530 | **0.755** | **0.623** |
| XGBoost | Class Weights | 0.849 | 0.841 | **0.673** | 0.508 | 0.579 |
| XGBoost | SMOTE | 0.849 | 0.841 | 0.615 | 0.581 | 0.598 |
| LightGBM | SMOTE | 0.840 | 0.838 | 0.635 | 0.530 | 0.578 |
| LightGBM | Class Weights | 0.839 | 0.835 | 0.531 | 0.731 | 0.615 |

**Key Findings:**
- **CatBoost with class weights** achieved best overall performance (ROC-AUC: 0.843, F1: 0.623)
- **Class weight strategies** generally outperformed SMOTE across all models
- **Top predictive features**: Contract type, tenure, monthly charges (via SHAP analysis)

## Installation & Usage

```bash
# Clone repository
git clone https://github.com/mburakmeric/xgb-lgbm-cat-Churn.git
cd xgb-lgbm-cat-Churn

# Install dependencies
pip install -r requirements.txt

# Run analysis
jupyter notebook notebooks/churn_boosting.ipynb
```

## Project Structure

```
churn/
├── data/
│   ├── churn.csv                    # Customer dataset (7,044 records)
│   └── model_comparison_results.csv # Model evaluation results
├── notebooks/
│   └── churn_boosting.ipynb        # Complete analysis workflow
├── requirements.txt                 # Dependencies
└── README.md                       # Documentation
```

## Technical Skills Demonstrated

- **Machine Learning**: Gradient boosting algorithms, imbalanced dataset handling, hyperparameter optimization
- **Data Science**: EDA, feature engineering, statistical validation, model interpretability
- **Tools**: XGBoost, LightGBM, CatBoost, SHAP, scikit-learn, pandas, matplotlib
- **Best Practices**: Cross-validation, pipeline development, reproducible analysis

## Dependencies

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
shap>=0.40.0
plotly>=5.0.0
jupyter>=1.0.0
```

---

**Author**: Mehmet Burak Meric  
**License**: MIT License

