import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Path to the directory containing your image subfolders
base_directory = "/Users/josna/Documents/GitHub/batdetect/data/output"  # Update this path

# List of class names
class_names = ["B", "C", "D", "E", "F", "SMU"]

# Load and preprocess images
X = []
y = []
image_height = 224  # Adjust based on your image dimensions
image_width = 224
channels = 3  # RGB images

for class_idx, class_name in enumerate(class_names):
    class_directory = os.path.join(base_directory, class_name)
    image_files = [file for file in os.listdir(class_directory) if file.endswith(".png")]
    
    for image_file in image_files:
        image_path = os.path.join(class_directory, image_file)
        image = Image.open(image_path).resize((image_width, image_height))
        image_array = np.array(image)
        # Preprocess the image if needed (e.g., normalization)
        X.append(image_array)
        y.append(class_idx)

# Convert lists to arrays
X = np.array(X)
y = np.array(y)

# Normalize pixel values to the range [0, 1]
X = X / 255.0

# Create a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
batch_size = 32
epochs = 10
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)