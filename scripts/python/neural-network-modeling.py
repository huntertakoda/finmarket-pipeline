import subprocess
import sys

def ensure_setuptools():
    try:
        # check if setuptools is available
        import setuptools
    except ModuleNotFoundError:
        print("Setuptools not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])
        print("Setuptools installed successfully.")

ensure_setuptools()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# loading 

file_path = "enhanced_financial_data.csv"
df = pd.read_csv(file_path, parse_dates=["Timestamp"])
df = df.dropna()

# ftre engineering

features = ["Open", "High", "Low", "Volume", "Volatility", "Close_7D_MA", "Close_30D_MA"]
target = "Close"

X = df[features]
y = df[target]

# scale features for neural network

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train-test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# (defining the neural network architecture)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2), 
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1) 
])

# (compiling the model)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# (training the model)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, 
                    validation_split=0.2, 
                    epochs=100, 
                    batch_size=32, 
                    callbacks=[early_stopping], 
                    verbose=1)

# (evaluating the model)

nn_predictions = model.predict(X_test).flatten()
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)

print(f"Neural Network - RMSE: {nn_rmse:.4f}, MAE: {nn_mae:.4f}, R²: {nn_r2:.4f}")

# (plotting training and validation loss)

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# (plotting actual vs predicted close prices)

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Close Price", color="blue")
plt.plot(nn_predictions, label="NN Predictions", linestyle="--", color="orange")
plt.title("Actual vs Predicted Close Prices (Neural Network)")
plt.xlabel("Sample")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()
