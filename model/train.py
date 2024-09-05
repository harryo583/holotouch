"""
This script trains a neural network for gesture classification using hand landmark data.

Steps:
    1. Loads the data from a CSV file.
    2. Prepares the data by separating features and labels, and converting labels to categorical format.
    3. Splits the data into training and test sets.
    4. Builds and compiles a neural network model.
    5. Trains the model and evaluates its performance.
    6. Saves the trained model to a file.
    7. Plots training and validation loss and accuracy.

Dependencies:
    - Pandas (pd)
    - TensorFlow (tf)
    - Scikit-learn (sklearn)
    - Matplotlib.pyplot (plt)
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data from a CSV file
data = pd.read_csv(
    "/Users/hongyiwang/Desktop/Projects/Gesture Recognition/model/hand_landmarks.csv")

# Separate features and labels
X = data.drop(columns=['class'])
y = data['class']

# Convert labels to categorical format
y_categorical = tf.keras.utils.to_categorical(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42)

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and capture the history
history = model.fit(X_train, y_train, epochs=500,
                    batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save(
    "/Users/hongyiwang/Desktop/Projects/Gesture Recognition/model/hand_landmarks_model.h5")

# Plot training history
plt.figure(figsize=(12, 5))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
