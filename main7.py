import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier  #RF image

# Path to the directory containing your spectrogram image files
image_directory = "/Users/josna/Documents/GitHub/batdetect/data/output"

# List all image files in the directory
image_files = [file for file in os.listdir(image_directory) if file.endswith(".png")]

# Placeholder labels (since you don't have actual labels)
y = np.zeros(len(image_files))  # You can adjust this placeholder label as needed

# Load your trained RandomForest model
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
# Load the trained model weights or use your existing model

# Batch size for processing images
batch_size = 32

# Iterate over images in batches
for batch_start in range(0, len(image_files), batch_size):
    batch_end = min(batch_start + batch_size, len(image_files))
    batch_image_files = image_files[batch_start:batch_end]

    # Load and preprocess batch of images
    batch_X = [load_and_preprocess_image(os.path.join(image_directory, file)) for file in batch_image_files]
    batch_X = np.array(batch_X)

    # Fit the model with batch of images (as there are no actual labels)
    random_forest.fit(batch_X, y[batch_start:batch_end])

    # Make predictions on the batch of images
    batch_y_pred = random_forest.predict(batch_X)

    # Print the predicted labels (assuming you have a label mapping)
    batch_predicted_species = [your_species_mapping[index] for index in batch_y_pred]
    print("Predicted Species (Batch):", batch_predicted_species)
