# Calorie Expenditure Prediction with Machine Learning

This project explores the use of supervised learning regression models to preduct calorie expenditure during physical activity using wearable sensor and demographic data. Two models - a Decision Tree Regressor and a Multilayer Perceptron (MLP) - are implemented, tuned, and evaluated side-by-side to compare accuracy, interpretability, and computational cost.

## Overview
Accurately estimating calories burned during exercise is difficult using traditional formulas, but modern wearable devices provide rich physiological data that enables data-driven prediction. This project evaluates how well different regression models can predict calories burned using features such as duration, heart rate, age, height, weight, sex, and body temperature.

The project focuses on:
- Comparing a simple, interpretable model (Decision Tree) against a more expressive model (MLP)
- Hyperparameter tuning using cross-validation
- Analyzing performance tradeoffs between accuracy and training time

## Dataset
The dataset is from the Kaggle competition **"Predict Calorie Expenditure"** and contains approximately **750,000 samples**.

**Features:**
- Sex (one-hot encoded)
- Age
- Height
- Weight
- Exercise duration
- Heart rate
- Body temperature

**Target:**
- Calories burned

> **Note:** The dataset is not included in this repository.
> To run the code, download the dataset from Kaggle and place 'train.csv' inside a 'data/' directory.

## Methods
Two regression models were implemented in Python using sickit-learn:

### Decision Tree Regressor
- No feature scaling required
- Hyperparameters tuned with GridSearchCV:
    - 'max_depth'
    - 'min_samples_leaf'
    - 'min_samples_split'
    - 'max_features'
- Provides built-in feature importance for interpretability

### Multilayer Perceptron (MLP) Regressor
- Implemented using a pipeline with 'StandardScaler'
- Hyperparameters tuned with GridSearchCV:
    - Hidden layer sizes
    - L2 regularization ('alpha')
    - Learning rate
    - Maximum iterations
- Early stopping enabled to reduce overfitting
- Permutation importance used for feature analysis

Both models use **5-fold cross validation** on an 80/20 train-test split with a fixed random seed for reproducibility.

## Evaluation Metrics
Models were evaluated using:
- **Mean Absolute Error (MAE)** - primary metric for tuning
- **Root Mean Squared Logarithmic Error (RMSLE)** - secondary metric for final evaluation

Learning curves, validation curves, feature importance plots, and training/inference timing were generated for both models.

## Results
| Model         | Test MAE | Training Time |
|--------------|----------|---------------|
| Decision Tree | ~2.47    | ~75 seconds   |
| MLP           | ~2.14    | ~469 seconds  |

Key findings:
- The MLP achieved slightly better accuracy but required significantly longer training time
- The Decision Tree trained and inferrred much faster while remaining competitive
- Exercise duration was the most important feature for both models
- Heart rate and age were more influential in the MLP than in the Decision Tree

## Repository Structure
.
├── src/
│ ├── decisionTree.py
│ └── mlp.py
├── report/
│ └── Final_Project_Report.pdf
├── figures/
│ └── (generated plots)
├── data/
│ └── README.md
├── requirements.txt
└── README.md

## How to Run
1. Clone the repository
2. Install dependencies:
    pip install -r requirements.txt
3. Download the Kaggle dataset and place 'train.csv' in 'data/'
4. Run either model:
    python src/decisionTree.py
    python src/mlp.py

Each script performs preprocessing, hyperparameter turing, evaluation, and generates plots.

## Technologies Used
- Python 3.11
- NumPy, Pandas
- sickit-learn
- Matplotlib

## Author
**Miles Anderson**
University of Oregon