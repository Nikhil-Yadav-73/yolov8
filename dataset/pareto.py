import os
import shutil
import random

# Paths
images_folder = "./license_plates/images/"
labels_folder = "./license_plates/labels/"
train_images_folder = "./license_plates/images/train/"
val_images_folder = "./license_plates/images/val/"
train_labels_folder = "./license_plates/labels/train/"
val_labels_folder = "./license_plates/labels/val/"

# Create folders
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))]

# Shuffle and split
random.shuffle(image_files)
split_index = int(len(image_files) * 0.8)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Move files
for file in train_files:
    shutil.move(os.path.join(images_folder, file), train_images_folder)
    label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
    if os.path.exists(os.path.join(labels_folder, label_file)):
        shutil.move(os.path.join(labels_folder, label_file), train_labels_folder)

for file in val_files:
    shutil.move(os.path.join(images_folder, file), val_images_folder)
    label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
    if os.path.exists(os.path.join(labels_folder, label_file)):
        shutil.move(os.path.join(labels_folder, label_file), val_labels_folder)

print("Dataset split into train and val!")