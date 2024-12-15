import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# loading

file_path = "enhanced_financial_data.csv"
df = pd.read_csv(file_path, parse_dates=["Timestamp"])
df = df.dropna()

# (feature selection and scaling)

features = ["Open", "High", "Low", "Volume", "Volatility", "Close_7D_MA", "Close_30D_MA"]
target = "Close"

X = df[features]
y = df[target]

# scale the features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train / test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# (base models: stacked ensemble setup)

catboost_model = CatBoostRegressor(verbose=0, random_state=42)
extratrees_model = ExtraTreesRegressor(n_estimators=100, random_state=42)

# k-fold stacking

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

# (meta-model for stacked ensemble)

meta_model = LinearRegression()
meta_model.fit(train_meta_features, y_train)
stacked_ensemble_predictions = meta_model.predict(test_meta_features)

# (neural network predictions)

nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
nn_model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

nn_predictions = nn_model.predict(X_test).flatten()

# (extra stacked ensemble: combining predictions)

meta_features = np.column_stack((stacked_ensemble_predictions, nn_predictions))
extra_meta_model = LinearRegression()
extra_meta_model.fit(meta_features, y_test)

# (final predictions)

extra_stacked_predictions = extra_meta_model.predict(meta_features)

# (evaluation)

extra_rmse = np.sqrt(mean_squared_error(y_test, extra_stacked_predictions))
extra_mae = mean_absolute_error(y_test, extra_stacked_predictions)
extra_r2 = r2_score(y_test, extra_stacked_predictions)

print(f"Extra Stacked Ensemble - RMSE: {extra_rmse:.4f}, MAE: {extra_mae:.4f}, R²: {extra_r2:.4f}")

# (plot actual vs predicted close prices)

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Close Price", color="blue")
plt.plot(extra_stacked_predictions, label="Extra Stacked Ensemble Predictions", linestyle="--", color="green")
plt.title("Actual vs Predicted Close Prices (Extra Stacked Ensemble)")
plt.xlabel("Sample")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()
