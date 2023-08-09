import os
import shutil

# Path to the directory containing your images
image_directory = "/Users/josna/Documents/GitHub/batdetect/data/output"  # Update this path

# List of class prefixes
class_prefixes = ["B", "C", "D", "E", "F", "SMU"]

# Create subfolders for each class
for class_prefix in class_prefixes:
    class_directory = os.path.join(image_directory, class_prefix)
    os.makedirs(class_directory, exist_ok=True)

# List all image files in the directory
image_files = [file for file in os.listdir(image_directory) if file.endswith(".png")]

# Move images to appropriate class subfolders
for image_file in image_files:
    for class_prefix in class_prefixes:
        if image_file.startswith(class_prefix):
            src_path = os.path.join(image_directory, image_file)
            dst_path = os.path.join(image_directory, class_prefix, image_file)
            shutil.move(src_path, dst_path)
            break  # Move to the next image

print("Images have been arranged into class subfolders.")
