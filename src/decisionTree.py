import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, make_scorer
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv') #Read the data

#print(df.head()) #Print the first few lines of the data
#print()

df = pd.get_dummies(df, columns=['Sex'], drop_first=True).astype({'Sex_male': int}) #convert object type to something that can be used in a decision tree

X = df.drop(['id', 'Calories'], axis=1) #separate the id and Calories columns. id because it's non-predictive and Calories because its the target variable

y = df['Calories'] #y now stores the calories column

#Splitting the data
#80/20 train+val vs. test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    'max_depth': [None, 3, 5, 8, 10],
    'min_samples_leaf': [1, 2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['squared_error']
}

rmsle_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true, y_pred)), 
    greater_is_better=False
)

scoring = {
    'MAE': 'neg_mean_absolute_error',
    'RMSLE': rmsle_scorer
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring=scoring,
    refit='MAE',
    return_train_score=True,
    verbose=2,
    n_jobs=-1
)

start_train = time.perf_counter()

#Tune of 80% train+val
grid.fit(X_trainval, y_trainval)

train_time = time.perf_counter() - start_train
print(f"Total tuning and training time: {train_time:.1f} seconds")

#Feature-Importance Bar Chart
best_tree = grid.best_estimator_
importances = pd.Series(
    best_tree.feature_importances_,
    index=X_trainval.columns
).sort_values(ascending=False)

n = min(10, len(importances))

plt.figure()
plt.title("Top {} Feature Importances".format(n))
plt.bar(range(n), importances.iloc[:n])
plt.xticks(range(n), importances.index[:n], rotation=45, ha="right")
plt.ylabel("Importance")
plt.tight_layout()

plt.savefig('feature_importances_DT.png', dpi=300, bbox_inches='tight')
plt.show()
#====================================

#Validation Curve for a Single Hyperparamater
cv = pd.DataFrame(grid.cv_results_)
mask = cv['param_min_samples_leaf'] == 1
group = cv[mask].groupby('param_max_depth')
depths = []
maes = []
for depth, df_depth in group:
    depths.append(depth if depth is not None else max(depths or [0]) + 1)
    maes.append(-df_depth['mean_test_MAE'].min())

plt.figure()
plt.title("Validation MAE vs Tree Depth")
plt.plot(depths, maes, marker='o')
plt.xlabel("max_depth")
plt.ylabel("Validation MAE")
plt.xticks(depths)
plt.tight_layout()

plt.savefig('validation_curve_DT.png', dpi=300, bbox_inches='tight')
plt.show()
#=========================================

#Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    best_tree,
    X_trainval, y_trainval,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

train_mae = -train_scores.mean(axis=1)
val_mae = -val_scores.mean(axis=1)

plt.figure()
plt.title("Learning Curve (MAE)")
plt.plot(train_sizes, train_mae, label="Training MAE")
plt.plot(train_sizes, val_mae, label="Validation MAE")
plt.xlabel("Training examples")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()

plt.savefig('learning_curve_DT.png', dpi=300, bbox_inches='tight')
plt.show()
#=============================

#Fina evaluation
start_pred = time.perf_counter()
y_pred = best_tree.predict(X_test)
pred_time = time.perf_counter() - start_pred
print(f"Test-set prediction time: {pred_time:.3f} seconds")
print("Final Test MAE: ", mean_absolute_error(y_test, y_pred))
print("Final Test RMSLE: ", np.sqrt(mean_squared_log_error(y_test, y_pred)))

best_tree_params = grid.best_params_
best_tree_mae = mean_absolute_error(y_test, y_pred)

summary_df = pd.DataFrame([
    {
        'Model': 'Decision Tree',
        **best_tree_params,
        'Test MAE': best_tree_mae
    }
])
print("\nBest Hyperparameters and Test MAE:\n", summary_df)

results = pd.DataFrame(grid.cv_results_)
results['MAE_pos'] = -results['mean_test_MAE']
print("\nTop CV configurations by MAE:")
print(
    results
        .sort_values('MAE_pos')
        .loc[:, [
            'param_max_depth', 
            'param_min_samples_leaf',
            'param_min_samples_split',
            'param_max_features',
            'mean_test_MAE'
        ]]
        .head()
)

#Feature importances
importances = pd.Series(
    best_tree.feature_importances_, index=X_trainval.columns
).sort_values(ascending=False)
print("Top 10 features:\n", importances.head(10))
