import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# testing / training split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. linear regression model

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# evaluate linear regression

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)

print(f"Linear Regression - RMSE: {lr_rmse:.4f}, R²: {lr_r2:.4f}")

# 2. random forest model

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# evaluate random forest

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Random Forest - RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")

# hyperparameter tuning for random forest using grid search

print("Optimizing Random Forest Hyperparameters...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']  # Fixed 'auto'
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                           param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# best model

best_rf_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# evaluate optimized random forest

best_rf_predictions = best_rf_model.predict(X_test)
best_rf_rmse = np.sqrt(mean_squared_error(y_test, best_rf_predictions))
best_rf_r2 = r2_score(y_test, best_rf_predictions)

print(f"Optimized Random Forest - RMSE: {best_rf_rmse:.4f}, R²: {best_rf_r2:.4f}")

# plot actual vs predicted

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Close Price", color="blue")
plt.plot(lr_predictions, label="Linear Regression Predictions", linestyle="--", color="orange")
plt.plot(rf_predictions, label="Random Forest Predictions", linestyle="--", color="green")
plt.plot(best_rf_predictions, label="Optimized RF Predictions", linestyle="--", color="red")
plt.title("Actual vs Predicted Close Prices")
plt.xlabel("Sample")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()
