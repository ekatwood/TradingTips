# from chatGPT

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Create Labels (Future Price for Prediction)
labels = []
for i in range(len(prices) - window_size - prediction_step):
    labels.append(prices[i + window_size + prediction_step])

# Convert to NumPy array for model training
labels = np.array(labels)

# Image size to match input for the CNN
image_size = (64, 64)

def load_images(image_dir, image_size):
    image_list = []
    for i in range(len(labels)):  # Only load as many images as labels we have
        image_path = os.path.join(image_dir, f'image_{i}.png')
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img_array = img_to_array(img)
            image_list.append(img_array)
    return np.array(image_list)

# Load images and resize them
X_images = load_images(image_dir, image_size)

# Normalize pixel values between 0 and 1
X_images = X_images / 255.0

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_images, labels, test_size=0.2, random_state=42)

# Build CNN Model
model = Sequential()

# 1st Convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output and add Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))  # Output layer (1 output for price prediction)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Model Summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict the prices using the test set
predicted_prices = model.predict(X_test)

# Inverse the scaling to get actual price predictions
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test)

# Plot the actual vs predicted prices
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.show()

# Print a few predictions vs actual values
for i in range(5):
    print(f"Predicted Price: {predicted_prices[i][0]:.2f}, Actual Price: {actual_prices[i][0]:.2f}")
