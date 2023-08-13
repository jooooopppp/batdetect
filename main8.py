import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import librosa

# Path to the directory containing your audio class subdirectories
base_directory = "/Users/josna/Documents/GitHub/batdetect/data/raw"  # Update this path

# List of class names (subdirectory names)
class_names = ["B" ]

# Load and preprocess audio data 
X = []
y = []
sample_rate = 44100  # Adjust based on your audio data's sample rate
duration = 2  # Adjust based on your desired audio segment duration
input_shape = (sample_rate * duration, 1)  # One channel (mono audio)

for class_idx, class_name in enumerate(class_names):
    class_directory = os.path.join(base_directory, class_name)
    audio_files = [file for file in os.listdir(class_directory) if file.endswith(".wav")]
    
    for audio_file in audio_files:
        audio_path = os.path.join(class_directory, audio_file)
        audio_data, _ = librosa.load(audio_path, sr=sample_rate, duration=duration)
        # Preprocess the audio data if needed (e.g., normalization)
        X.append(audio_data)
        y.append(class_idx)

# Convert lists to arrays
X = np.array(X)
y = np.array(y)

# Normalize audio data if needed
X = X / np.max(np.abs(X))

# Convert class labels to one-hot encoded format
y = to_categorical(y, num_classes=len(class_names))

# Create a CNN model for audio
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=input_shape),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
batch_size = 32
epochs = 10
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Plot training accuracy and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()  # Display the plot

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)