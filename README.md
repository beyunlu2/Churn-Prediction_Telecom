# Telecom Customer Churn Prediction

## Overview

This project implements a machine learning solution to predict customer churn in the telecommunications industry. Customer churn prediction is crucial for telecom companies to identify customers who are likely to cancel their services, enabling proactive retention strategies.

## Dataset

The project uses the `telecom_customer_churn.csv` dataset containing customer information and service usage patterns with the following characteristics:

- **Total Records**: 7,043 customers
- **Features**: 38 original columns including customer demographics, service usage, billing information, and churn status
- **Target Variable**: Customer Status (Stayed/Churned)

### Key Features Include:
- **Demographics**: Age, Gender, Marital Status, Number of Dependents
- **Location**: City, Zip Code, Latitude, Longitude
- **Services**: Phone Service, Internet Service, Multiple Lines, Streaming Services
- **Contract Information**: Contract Type, Payment Method, Paperless Billing
- **Financial**: Monthly Charges, Total Charges, Total Revenue, Refunds

## Data Preprocessing

### 1. Data Cleaning
- Removed columns with excessive missing values: `Offer`, `Churn Category`, `Churn Reason`
- Converted target variable to binary classification (Stayed: 0, Churned: 1)
- Excluded "Joined" class to focus on binary classification

### 2. Missing Value Treatment
- **Numerical Features**:
  - `Avg Monthly Long Distance Charges`: Filled with mean (normal distribution)
  - `Avg Monthly GB Download`: Filled with median (skewed distribution)
- **Categorical Features**: Missing values filled with "No" for service-related columns

### 3. Feature Engineering
- **Dropped irrelevant features**: Customer ID, City (too many unique values), Internet Type
- **Label Encoding**: Applied to binary categorical variables (Yes/No → 1/0)
- **One-Hot Encoding**: Applied to multi-category variables (Gender, Contract, Payment Method)
- **Feature Combination**: Created `Service Count` by combining multiple service columns
- **Customer Value**: Calculated as `Total Revenue / Tenure in Months`

### 4. Feature Selection
- Used Random Forest feature importance to identify key predictors
- Removed low-importance features to reduce dimensionality
- Handled multicollinearity by removing highly correlated features (correlation > 0.8)

### 5. Data Normalization
- Applied StandardScaler for algorithms sensitive to feature scaling
- Removed outliers using IQR method for relevant numerical features

## Machine Learning Models

The project implements and compares three different algorithms:

### 1. XGBoost Classifier
- **Hyperparameter Optimization**: Using Optuna with 50 trials
- **Best Performance**: 
  - Accuracy: 82.72%
  - F1-Score: 78.59%
  - AUC-ROC: 90.29%

### 2. Random Forest Classifier
- **Hyperparameter Optimization**: Using Optuna with 50 trials
- **Performance**: 
  - Accuracy: 81.88%
  - F1-Score: 77.87%
  - AUC-ROC: 89.27%

### 3. Logistic Regression
- **Hyperparameter Optimization**: Using Optuna with 100 trials
- **Performance**: 
  - Accuracy: 76.34%
  - F1-Score: 71.40%
  - AUC-ROC: 82.81%

## Key Findings

### Most Important Features for Churn Prediction:
1. **Total Long Distance Charges** (15.36% importance)
2. **Contract Month-to-Month** (11.91% importance)
3. **Age** (9.88% importance)
4. **Monthly Charge** (9.73% importance)
5. **Number of Referrals** (8.21% importance)

### Model Performance Summary:
- **Best Model**: XGBoost with 90.29% AUC-ROC score
- **Most Balanced**: Random Forest with good accuracy-precision trade-off
- **Baseline**: Logistic Regression provides interpretable results

## Business Implications

1. **Contract Type**: Month-to-month contracts show highest churn risk
2. **Pricing Strategy**: High monthly charges correlate with increased churn
3. **Customer Engagement**: Customers with more referrals tend to stay longer
4. **Service Usage**: Long-distance charges are a strong churn indicator
5. **Demographics**: Age plays a significant role in churn behavior

## Requirements

```python
# Core Libraries
pandas
numpy
matplotlib
seaborn

# Machine Learning
scikit-learn
xgboost
optuna

# Environment
Google Colab (for file access)
```

## Usage

1. **Data Loading**: Mount Google Drive and load the dataset
2. **Preprocessing**: Run data cleaning and feature engineering cells
3. **Model Training**: Execute hyperparameter optimization for all models
4. **Evaluation**: Compare model performances using test set metrics
5. **Visualization**: Generate ROC curves and confusion matrices

## Files Structure

```
deneme/
├── Churn_Prediction_Telecom.ipynb  # Main notebook
└── README_Churn_Prediction_Telecom.md  # This documentation
```

## Evaluation Metrics

The models are evaluated using multiple metrics:
- **Accuracy**: Overall prediction correctness
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions vs actual

## Future Improvements

1. **Feature Engineering**: Create more sophisticated customer behavior features
2. **Ensemble Methods**: Combine multiple models for better performance
3. **Real-time Prediction**: Implement streaming prediction pipeline
4. **Customer Segmentation**: Apply clustering for targeted retention strategies
5. **Time Series Analysis**: Incorporate temporal patterns in customer behavior

## Conclusion

The XGBoost model achieved the best performance with 90.29% AUC-ROC, making it suitable for production deployment. The analysis reveals that contract type, pricing, and customer engagement are the most critical factors in predicting churn, providing actionable insights for business strategy. 
