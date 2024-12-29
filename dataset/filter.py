import os

# Paths to your dataset directories
images_val_dir = r"C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\license_plates\images\val"
labels_val_dir = r"C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\license_plates\labels\val"

def remove_unlabeled_images(images_dir, labels_dir):
    # Get lists of image and label filenames (without extensions)
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}
    
    # Find images without labels
    images_without_labels = image_files - label_files

    # Remove images without labels
    for img in images_without_labels:
        img_path = os.path.join(images_dir, img + '.jpg')  # Change extension if needed
        if not os.path.exists(img_path):  # Handle cases where the extension is different
            img_path = os.path.join(images_dir, img + '.png')
        if os.path.exists(img_path):
            print(f"Removing image without label: {img_path}")
            os.remove(img_path)

# Process validation directory
print("Processing validation set...")
remove_unlabeled_images(images_val_dir, labels_val_dir)

print("Cleanup #2 completed.")