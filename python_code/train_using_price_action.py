# from chatGPT, requires versioning (this is like a first draft)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load historical price data (time, open, high, low, close, volume)
# Example data can be a CSV file with historical data, here 'crypto_price_data.csv'
data = pd.read_csv('crypto_price_data.csv', parse_dates=['timestamp'])
data.set_index('timestamp', inplace=True)

# Resample data into 3-hour intervals
data_resampled = data.resample('3H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Use only the 'close' price for simplicity in this example
prices = data_resampled['close'].values

# Prepare input (X) and output (y)
# X will be the last 3 hours of prices (a window of size 3)
# y will be the price at some future point in the next 3 hours (e.g., 3 hours ahead)

def create_sequences(data, window_size, prediction_step):
    X = []
    y = []
    for i in range(len(data) - window_size - prediction_step):
        X.append(data[i:i + window_size])  # Input: 3-hour window
        y.append(data[i + window_size + prediction_step])  # Output: price at 'prediction_step' ahead
    return np.array(X), np.array(y)

# Define window size (last 3 hours) and prediction step (next 3 hours)
window_size = 1  # We're using 3-hour intervals
prediction_step = 1  # Predicting 3 hours ahead

# Prepare the sequences
X, y = create_sequences(prices, window_size, prediction_step)

# Reshape X to be [samples, time steps, features] for LSTM input
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and test sets (80% train, 20% test)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))  # Predict a single value (future price)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Invert scaling to get the actual price
predicted_prices = scaler.inverse_transform(predictions)

# Get actual test prices
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Output a few predictions and actual prices for comparison
for i in range(5):
    print(f"Predicted price: {predicted_prices[i][0]:.2f}, Actual price: {actual_prices[i][0]:.2f}")
