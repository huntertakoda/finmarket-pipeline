import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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

# scale the features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train-test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# initialize and train the svr model

svr_model = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
svr_model.fit(X_train, y_train)

# predictions

svr_predictions = svr_model.predict(X_test)

# evaluate svr performance

svr_rmse = np.sqrt(mean_squared_error(y_test, svr_predictions))
svr_r2 = r2_score(y_test, svr_predictions)

print(f"SVR - RMSE: {svr_rmse:.4f}, R²: {svr_r2:.4f}")

# hyperparameter tuning for svr using grid search

print("optimizing svr hyperparameters...")
param_grid = {
    'C': [1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(SVR(), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# best model

best_svr_model = grid_search.best_estimator_
print("best hyperparameters:", grid_search.best_params_)

# evaluate optimized svr

best_svr_predictions = best_svr_model.predict(X_test)
best_svr_rmse = np.sqrt(mean_squared_error(y_test, best_svr_predictions))
best_svr_r2 = r2_score(y_test, best_svr_predictions)

print(f"optimized svr - RMSE: {best_svr_rmse:.4f}, R²: {best_svr_r2:.4f}")

# plot actual vs predicted

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="actual close price", color="blue")
plt.plot(svr_predictions, label="svr predictions", linestyle="--", color="orange")
plt.plot(best_svr_predictions, label="optimized svr predictions", linestyle="--", color="red")
plt.title("actual vs predicted close prices (svr)")
plt.xlabel("sample")
plt.ylabel("close price")
plt.legend()
plt.tight_layout()
plt.show()
