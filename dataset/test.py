import os

def check_image_label_matching(image_dir, label_dir):
    images = {f.split('.')[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}
    labels = {f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

    missing_labels = images - labels
    if missing_labels:
        print("Missing labels for the following images:")
        for img in missing_labels:
            print(img)
    else:
        print("All images have corresponding labels.")

# Check if every image has a corresponding label in train and val directories
check_image_label_matching(r'C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\license_plates\images\val', r'C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\license_plates\labels\val')
check_image_label_matching(r'C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\license_plates\images\train', r'C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\license_plates\labels\train')
