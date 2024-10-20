# from chatGPT, interpolated functions can be found in interpolated_objects/

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load Historical Cryptocurrency Data
# Example CSV file: [timestamp, open, high, low, close, volume]
data = pd.read_csv('crypto_price_data.csv', parse_dates=['timestamp'])
data.set_index('timestamp', inplace=True)

# Use 'close' prices for this example
prices = data['close'].values
timestamps = data.index.astype('int64') // 1e9  # Convert to Unix timestamp for interpolation

# 2. Interpolate the Data with Cubic Spline
cs = CubicSpline(timestamps, prices)

# Generate new timestamps for interpolation (more frequent, e.g., every 5 minutes within 3 hours)
new_timestamps = np.linspace(timestamps.min(), timestamps.max(), len(timestamps) * 5)
interpolated_prices = cs(new_timestamps)

# 3. Normalize the Interpolated Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_interpolated_prices = scaler.fit_transform(interpolated_prices.reshape(-1, 1))

# 4. Prepare Input and Output Sequences
def create_sequences(data, window_size, prediction_step):
    X = []
    y = []
    for i in range(len(data) - window_size - prediction_step):
        X.append(data[i:i + window_size])  # Input: 3-hour window of interpolated prices
        y.append(data[i + window_size + prediction_step])  # Output: predicted price in next 3 hours
    return np.array(X), np.array(y)

# Define window size (e.g., 3-hour window of interpolated data points) and prediction step
window_size = 36  # Assume 12 data points per hour after interpolation (e.g., every 5 minutes)
prediction_step = 12  # Predict 1 hour ahead (can adjust as needed)

# Create input and output sequences
X, y = create_sequences(scaled_interpolated_prices, window_size, prediction_step)

# Reshape input for LSTM: [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# 5. Split Data into Training and Testing Sets
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 6. Define the LSTM Model
model = Sequential()

# Add LSTM layer with 64 units
model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train.shape[1], 1)))

# Output layer to predict future price
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 8. Make Predictions on the Test Set
predictions = model.predict(X_test)

# Inverse the scaling to get the actual prices
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Print a few predictions for comparison
for i in range(5):
    print(f"Predicted Price: {predicted_prices[i][0]:.2f}, Actual Price: {actual_prices[i][0]:.2f}")
