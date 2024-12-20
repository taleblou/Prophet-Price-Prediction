
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score
import yfinance as yf
from tqdm import tqdm
import os
from prophet import Prophet
import tensorflow as tf
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Check TensorFlow version and GPU availability
print("TensorFlow Version:", tf.__version__)
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Device handling
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f"Using device: {device}")

# Utility for logging
def text_write(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text + "\n")

# Load data from Yahoo Finance
def load_data(ticker):
    data = yf.download(ticker)
    return data


# Configuration
name = "BTC-USD"# "GC=F", "EURUSD=X", "^GSPC"
file_path = f"prophet_{name}_log.txt"

# Load and preprocess data
data = load_data(name)
data = data[["Close", "Open", "High", "Low"]][-1000:]  # Focus on the last 1000 records
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

# Prepare targets
for col in ["Close", "Open", "High", "Low"]:
    data[f"y_{col}"] = data[col].shift(-1)
data.dropna(inplace=True)

# Features and targets
X = data[["Close", "Open", "High", "Low"]].values
Y = data[["y_Close", "y_Open", "y_High", "y_Low"]].values

# Add prediction columns
for col in ["Close", "Open", "High", "Low"]:
    data[f"p_{col}"] = np.nan
    data[f"o_p_{col}"] = np.nan
    data[f"o_y_{col}"] = np.nan

# Rolling prediction
box = 200
for i in tqdm(range(box, len(data))):
    X_train = X[:i - box]
    Y_train = Y[:i - box]
    X_test = X[i - box:i - box + 1]
    Y_test = Y[i - box:i - box + 1]

    for c, col in enumerate(["Close", "Open", "High", "Low"]):
        # Prepare training data for Prophet
        train_data = pd.DataFrame({
            'ds': data.index[:i - box],  # Use timestamps for 'ds'
            'y': Y_train[:, c]          # Use the corresponding target column for 'y'
        })

        # Skip iteration if not enough data
        if train_data.empty or len(train_data) < 2:
            continue

        # Train Prophet model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(train_data)

        # Create future DataFrame for predictions
        future = pd.DataFrame({'ds': data.index[i - box:i - box + 1]})
        forecast = model.predict(future)

        # Extract prediction
        prediction = forecast['yhat'].iloc[0]
        data.loc[data.index[i - box + 1], f"p_{col}"] = prediction

        # Scale back predictions and targets to original range
        scaled_predictions = np.zeros((1, 4))
        scaled_predictions[0, c] = prediction
        inverse_predictions = scaler.inverse_transform(scaled_predictions)
        inverse_targets = scaler.inverse_transform(Y_test)

        data.loc[data.index[i - box + 1], f"o_p_{col}"] = inverse_predictions[0, c]
        data.loc[data.index[i - box + 1], f"o_y_{col}"] = inverse_targets[0, c]

# Save predictions
data.to_csv(f"Predictions_{name}.csv")

data.dropna(inplace=True)
# Evaluate and visualize
for col in ["Close", "Open", "High", "Low"]:
    actual = data[f"y_{col}"].dropna()
    predicted = data[f"p_{col}"].dropna()

    # Metrics
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    medae = median_absolute_error(actual, predicted)
    evs = explained_variance_score(actual, predicted)

    text_write(file_path, f"{col} Metrics - MSE: {mse}, MAE: {mae}, R2: {r2}, MedAE: {medae}, EVS: {evs}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data[f"o_y_{col}"], label=f"Actual {col}", color='blue')
    plt.plot(data.index, data[f"o_p_{col}"], label=f"Predicted {col}", color='green')
    plt.title(f"{col} Price Prediction")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"prophet_{name}_{col}.png")
    plt.close()
