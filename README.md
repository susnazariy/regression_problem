# California Housing Price Prediction

A machine learning project for predicting median house values in California using the California Housing dataset. This project explores multiple regression algorithms and implements hyperparameter tuning to find the optimal model.

## Overview

This project aims to predict housing prices based on various geographical and demographic features. It follows a complete ML pipeline from data cleaning to model comparison and selection.

## Dataset

The California Housing dataset contains **20,640 samples** with the following features:

| Feature | Description |
|---------|-------------|
| `longitude` | Geographic coordinate (west direction) |
| `latitude` | Geographic coordinate (north direction) |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` | Total rooms in the block |
| `total_bedrooms` | Total bedrooms in the block |
| `population` | Block population |
| `households` | Number of households in the block |
| `median_income` | Median household income (in $10,000s) |
| `ocean_proximity` | Categorical: location relative to ocean |
| **`median_house_value`** | **Target variable** (in USD) |

## Project Pipeline

### 1. Data Cleaning
- Handled missing values in `total_bedrooms` column
- Removed outliers using IQR method

### 2. Exploratory Data Analysis
- Geographic visualization of house values and income distribution
- Correlation analysis between features
- Distribution analysis of numerical features

### 3. Feature Engineering
- **One-Hot Encoding**: Converted `ocean_proximity` categorical feature
- **PCA**: Applied Principal Component Analysis on highly correlated features (`total_rooms`, `total_bedrooms`, `population`, `households`) to address multicollinearity

### 4. Model Training & Comparison

Implemented a custom `Regressor` class to streamline model training and evaluation. Models tested:

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Ridge | L2 regularization |
| Lasso | L1 regularization |
| ElasticNet | Combined L1/L2 regularization |
| Decision Tree | Tree-based regressor |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential ensemble method |
| KNeighbors | Instance-based learning |
| AdaBoost | Adaptive boosting |

### 5. Hyperparameter Tuning
- Used `GridSearchCV` for systematic hyperparameter optimization
- Compared default vs. tuned model performance

## Results

**Best Model: Random Forest Regressor**

| Metric | Train | Test |
|--------|-------|------|
| R² Score | 0.972 | 0.804 |
| RMSE | $15,941 | $42,579 |

Optimal hyperparameters: `n_estimators=200`, `max_depth=None`

## Tech Stack

- **Python 3.13**
- **pandas** — Data manipulation
- **NumPy** — Numerical computing
- **scikit-learn** — ML algorithms & preprocessing
- **Matplotlib & Seaborn** — Visualization

## Project Structure

```
├── CaliforniaHousingML.ipynb   # Main notebook with complete analysis
├── housing.csv                  # Dataset
└── README.md                    # Project documentation
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/california-housing-ml.git
   cd california-housing-ml
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook CaliforniaHousingML.ipynb
   ```

## Key Findings

- **Median income** is the strongest predictor of house prices
- **Geographic location** (longitude/latitude) significantly impacts housing values
- **Coastal proximity** correlates with higher property values
- **Ensemble methods** (Random Forest, Gradient Boosting) outperform linear models
- Addressing multicollinearity via PCA improved model stability

## License

This project is open source and available under the [MIT License](LICENSE).
