import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

# loading

file_path = "enhanced_financial_data.csv"
df = pd.read_csv(file_path, parse_dates=["Timestamp"])

# drop missing values

df = df.dropna()

# feature engineering, select predictors and target

features = ["Open", "High", "Low", "Volume", "Volatility", "Close_7D_MA", "Close_30D_MA"]
target = "Close"

X = df[features]
y = df[target]

# scale features for better performance

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# testing / training split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. catboost model with built-in grid search

print("tuning catboost hyperparameters...")
train_pool = Pool(X_train, y_train)

catboost_model = CatBoostRegressor(random_state=42, verbose=0)

catboost_grid = {
    'iterations': [500, 1000],  # Higher iterations for precision
    'depth': [6, 8, 10],  # Depth tuning
    'learning_rate': [0.01, 0.05, 0.1]  # Learning rate tuning
}

# catboost's built-in grid search
catboost_model.grid_search(catboost_grid, train_pool, cv=3, verbose=2)

# best model predictions
catboost_predictions = catboost_model.predict(X_test)

# evaluate catboost

catboost_rmse = np.sqrt(mean_squared_error(y_test, catboost_predictions))
catboost_mae = mean_absolute_error(y_test, catboost_predictions)
catboost_r2 = r2_score(y_test, catboost_predictions)

print(f"catboost - rmse: {catboost_rmse:.4f}, mae: {catboost_mae:.4f}, r²: {catboost_r2:.4f}")

# 2. extratrees regressor with grid searchcv

print("tuning extratrees hyperparameters...")

extratrees_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

extratrees_model = GridSearchCV(
    estimator=ExtraTreesRegressor(random_state=42),
    param_grid=extratrees_param_grid,
    cv=3,
    n_jobs=-1,  # Use all cores efficiently
    verbose=2  # Show progress
)
extratrees_model.fit(X_train, y_train)
best_extratrees = extratrees_model.best_estimator_
extratrees_predictions = best_extratrees.predict(X_test)

# evaluate extratrees

extratrees_rmse = np.sqrt(mean_squared_error(y_test, extratrees_predictions))
extratrees_mae = mean_absolute_error(y_test, extratrees_predictions)
extratrees_r2 = r2_score(y_test, extratrees_predictions)

print(f"extratrees - rmse: {extratrees_rmse:.4f}, mae: {extratrees_mae:.4f}, r²: {extratrees_r2:.4f}")

# plot actual vs predicted

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="actual close price", color="blue", linewidth=1)
plt.plot(catboost_predictions, label="catboost predictions", linestyle="--", color="orange", linewidth=1)
plt.plot(extratrees_predictions, label="extratrees predictions", linestyle="--", color="red", linewidth=1)
plt.title("actual vs predicted close prices")
plt.xlabel("sample")
plt.ylabel("close price")
plt.legend()
plt.tight_layout()
plt.show()
