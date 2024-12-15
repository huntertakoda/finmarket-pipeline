import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# loading

file_path = "enhanced_financial_data.csv"
df = pd.read_csv(file_path, parse_dates=["Timestamp"])

# drop missing values

df = df.dropna()

# feature engineering

features = ["Open", "High", "Low", "Volume", "Volatility", "Close_7D_MA", "Close_30D_MA"]
target = "Close"

X = df[features]
y = df[target]

# train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. xgboost

xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# evaluate xgboost

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)

print(f"xgboost - rmse: {xgb_rmse:.4f}, r²: {xgb_r2:.4f}")

# 2. lightgbm

lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.fit(X_train, y_train)
lgbm_predictions = lgbm_model.predict(X_test)

# evaluate lightgbm

lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_predictions))
lgbm_r2 = r2_score(y_test, lgbm_predictions)

print(f"lightgbm - rmse: {lgbm_rmse:.4f}, r²: {lgbm_r2:.4f}")

# plotting actual vs predicted

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="actual close price", color="blue")
plt.plot(xgb_predictions, label="xgboost", linestyle="--", color="purple")
plt.plot(lgbm_predictions, label="lightgbm", linestyle="--", color="red")
plt.title("actual vs predicted close prices")
plt.xlabel("sample")
plt.ylabel("close price")
plt.legend()
plt.tight_layout()
plt.show()
