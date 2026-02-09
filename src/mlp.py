import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

df = pd.read_csv('data/train.csv')

df = pd.get_dummies(df, columns=['Sex'], drop_first=True).astype({'Sex_male': int})

X = df.drop(['id', 'Calories'], axis=1)
y = df['Calories']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    'mlp__hidden_layer_sizes': [(50, ), (100, ), (50,50), (100,50)],
    'mlp__alpha': [1e-4, 1e-3, 1e-2],
    'mlp__learning_rate_init': [1e-3, 1e-4],
    'mlp__max_iter': [300, 500],
}

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    ))
])

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    refit=True,
    return_train_score=True,
    verbose=2,
    n_jobs=-1
)

start_train = time.perf_counter()
grid.fit(X_trainval, y_trainval)
train_time = time.perf_counter() - start_train
print(f"Total tuning & training time: {train_time:.1f} seconds")

results = pd.DataFrame(grid.cv_results_)

print(
    results
        .sort_values('mean_test_score', ascending=False)
        .loc[:, ['param_mlp__hidden_layer_sizes',
                 'mean_test_score',]]
        .head()
)

#Feature-Importance Bar Chart
r = permutation_importance(grid.best_estimator_, X_trainval, y_trainval, n_repeats=10, random_state=42)
importances = pd.Series(r.importances_mean, index=X_trainval.columns).sort_values(ascending=False)

n = min(10, len(importances))

plt.figure()
plt.title(f"Top {n} Features (MLP, by Permutation Importance)")
plt.bar(range(n), importances.iloc[:n])
plt.xticks(range(n), importances.index[:n], rotation=45, ha='right')
plt.tight_layout()

plt.savefig('feature_importances_MLP.png', dpi=300, bbox_inches='tight')
plt.show()
#==============================

#Validation Curve for a single hyperparamater
cv = pd.DataFrame(grid.cv_results_)
mask = cv['param_mlp__alpha'] == 1e-3
group = cv[mask].groupby('param_mlp__hidden_layer_sizes')
sizes, maes = zip(*[
    (str(size), -df_size['mean_test_score'].min())
    for size, df_size in group
])

plt.figure()
plt.title("Validation MAE vs Hidden Layer Size (alpha=1e-3)")
plt.plot(sizes, maes, marker='o')
plt.xlabel("hidden_layer_sizes")
plt.ylabel("Validation MAE")
plt.tight_layout()

plt.savefig('validation_curve_MLP.png', dpi=300, bbox_inches='tight')
plt.show()
#==========================================

#Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    grid.best_estimator_,
    X_trainval, y_trainval,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

train_mae = -train_scores.mean(axis=1)
val_mae = -val_scores.mean(axis=1)

plt.figure()
plt.title("Learning Curve (MLP)")
plt.plot(train_sizes, train_mae, label="Training MAE")
plt.plot(train_sizes, val_mae, label="Validation MAE")
plt.xlabel("Training examples")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()

plt.savefig('learning_curve_MLP.png', dpi=300, bbox_inches='tight')
plt.show()
#=============================

#Loss Curve
best = grid.best_estimator_.named_steps['mlp']
plt.plot(best.loss_curve_)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('MLP Training Loss Curve')
plt.savefig('loss_curve_MLP.png', dpi=300, bbox_inches='tight')
plt.show()
#==========================

start_pred = time.perf_counter()
y_pred = grid.best_estimator_.predict(X_test)
pred_time = time.perf_counter() - start_pred
print(f"Test-set prediction time: {pred_time:.3f} seconds")
y_pred_clipped = np.maximum(0, y_pred)
print("Final Test MAE: ", mean_absolute_error(y_test, y_pred))
test_msle = mean_squared_log_error(y_test, y_pred_clipped)
test_rmsle = np.sqrt(test_msle)
print("Final Test RMSLE: ", test_rmsle)

best_mlp_params = grid.best_params_
best_mlp_mae = mean_absolute_error(y_test, y_pred)

summary_mlp = pd.DataFrame([{
    'Model': 'MLP',
    **best_mlp_params,
    'Test MAE': best_mlp_mae
}])
print("\nBEst Hyperparameters and TEst MAE (MLP):\n", summary_mlp)

results = pd.DataFrame(grid.cv_results_)

results['MAE_pos'] = -results['mean_test_score']

print("\nTop CV configurations by MAE (MLP):")
print(
    results
        .sort_values('MAE_pos')
        .loc[:, [
            'param_mlp__hidden_layer_sizes',
            'param_mlp__alpha',
            'param_mlp__learning_rate_init',
            'param_mlp__max_iter',
            'MAE_pos'
        ]]
        .head()
)