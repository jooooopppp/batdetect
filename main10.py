import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Path to the directory containing your image subfolders
base_directory = "/Users/josna/Documents/GitHub/batdetect/data/output"  # Update this path
class_names = ["B", "C", "D", "E", "F", "SMU"]

# Load and preprocess images
X = []
y = []
image_height = 224  # Adjust based on your image dimensions
image_width = 224
channels = 3  # RGB images (you can adjust channels based on your data)

for class_idx, class_name in enumerate(class_names):
    class_directory = os.path.join(base_directory, class_name)
    image_files = [file for file in os.listdir(class_directory) if file.endswith(".png")]
    
    for image_file in image_files:
        image_path = os.path.join(class_directory, image_file)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_height, image_width))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        X.append(image_array)
        y.append(class_idx)

X = np.array(X)
y = np.array(y)

# Normalize pixel values to the range [0, 1]
X = X / 255.0

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Add dropout to prevent overfitting
    Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create an image data generator with data augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model using generators
batch_size = 32
epochs = 10

# Train the model using generators and get history
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_test, y_test))

# Plot training accuracy and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test Accuracy:", test_acc)

# Show the plot
plt.show()





