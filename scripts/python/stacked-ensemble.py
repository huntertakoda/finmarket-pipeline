import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
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

# scale features for better performance

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# testing / training split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# base models

catboost_model = CatBoostRegressor(verbose=0, random_state=42)
extratrees_model = ExtraTreesRegressor(n_estimators=100, random_state=42)

# meta model

meta_model = LinearRegression()

# stacking: k-fold cross-validation for meta-model training

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_meta_features = np.zeros((X_train.shape[0], 2))
test_meta_features = np.zeros((X_test.shape[0], 2))

for train_idx, val_idx in kf.split(X_train):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # train catboost
    catboost_model.fit(X_fold_train, y_fold_train)
    train_meta_features[val_idx, 0] = catboost_model.predict(X_fold_val)
    test_meta_features[:, 0] += catboost_model.predict(X_test) / kf.n_splits
    
    # train extratrees
    extratrees_model.fit(X_fold_train, y_fold_train)
    train_meta_features[val_idx, 1] = extratrees_model.predict(X_fold_val)
    test_meta_features[:, 1] += extratrees_model.predict(X_test) / kf.n_splits

# train meta-model

meta_model.fit(train_meta_features, y_train)
stacked_predictions = meta_model.predict(test_meta_features)

# evaluate stacked model

stacked_rmse = np.sqrt(mean_squared_error(y_test, stacked_predictions))
stacked_mae = mean_absolute_error(y_test, stacked_predictions)
stacked_r2 = r2_score(y_test, stacked_predictions)

print(f"stacked model - rmse: {stacked_rmse:.4f}, mae: {stacked_mae:.4f}, r²: {stacked_r2:.4f}")

# evaluate base models

catboost_model.fit(X_train, y_train)
catboost_predictions = catboost_model.predict(X_test)
catboost_rmse = np.sqrt(mean_squared_error(y_test, catboost_predictions))
catboost_mae = mean_absolute_error(y_test, catboost_predictions)
catboost_r2 = r2_score(y_test, catboost_predictions)

extratrees_model.fit(X_train, y_train)
extratrees_predictions = extratrees_model.predict(X_test)
extratrees_rmse = np.sqrt(mean_squared_error(y_test, extratrees_predictions))
extratrees_mae = mean_absolute_error(y_test, extratrees_predictions)
extratrees_r2 = r2_score(y_test, extratrees_predictions)

print(f"catboost - rmse: {catboost_rmse:.4f}, mae: {catboost_mae:.4f}, r²: {catboost_r2:.4f}")
print(f"extratrees - rmse: {extratrees_rmse:.4f}, mae: {extratrees_mae:.4f}, r²: {extratrees_r2:.4f}")

# plot actual vs predicted

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="actual close price", color="blue", linewidth=1)
plt.plot(catboost_predictions, label="catboost predictions", linestyle="--", color="orange", linewidth=1)
plt.plot(extratrees_predictions, label="extratrees predictions", linestyle="--", color="red", linewidth=1)
plt.plot(stacked_predictions, label="stacked model predictions", linestyle="--", color="green", linewidth=1)
plt.title("actual vs predicted close prices")
plt.xlabel("sample")
plt.ylabel("close price")
plt.legend()
plt.tight_layout()
plt.show()

