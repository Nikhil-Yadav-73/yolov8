import os
import random
import xml.etree.ElementTree as ET
import shutil

# Define paths for your dataset
dataset_dir = r"C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\license_plates"
images_dir = os.path.join(dataset_dir, "images")
train_dir = os.path.join(images_dir, "train")
val_dir = os.path.join(images_dir, "val")
labels_dir = os.path.join(dataset_dir, "labels")
train_labels_dir = os.path.join(labels_dir, "train")
val_labels_dir = os.path.join(labels_dir, "val")
xml_dir = r"C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\xml_annotation"  # Path to your XML annotations

# Ensure directories exist
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

def convert_xml_to_yolo(xml_file, image_width, image_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    label_data = []
    
    # Process each object in the XML file
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        
        # Convert bounding box to YOLO format
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / float(image_width)
        height = (ymax - ymin) / float(image_height)
        
        # Write to YOLO format: <class_id> <x_center> <y_center> <width> <height>
        label_data.append(f"0 {x_center} {y_center} {width} {height}")
    
    return label_data

def create_yolo_labels():
    image_files = [f for f in os.listdir(train_dir) if f.endswith(".png") or f.endswith(".jpg")]
    
    for image_file in image_files:
        # Corresponding XML file
        xml_file = os.path.join(xml_dir, image_file.replace(".png", ".xml").replace(".jpg", ".xml"))
        
        if os.path.exists(xml_file):
            # Parse XML to get image dimensions and objects
            tree = ET.parse(xml_file)
            root = tree.getroot()
            image_width = int(root.find('size/width').text)
            image_height = int(root.find('size/height').text)
            
            label_data = convert_xml_to_yolo(xml_file, image_width, image_height)
            
            if label_data:
                label_file_path = os.path.join(train_labels_dir, image_file.replace(".png", ".txt").replace(".jpg", ".txt"))
                with open(label_file_path, 'w') as label_file:
                    label_file.write("\n".join(label_data))
            
    for image_file in os.listdir(val_dir):
        xml_file = os.path.join(xml_dir, image_file.replace(".png", ".xml").replace(".jpg", ".xml"))
        
        if os.path.exists(xml_file):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            image_width = int(root.find('size/width').text)
            image_height = int(root.find('size/height').text)
            
            label_data = convert_xml_to_yolo(xml_file, image_width, image_height)
            
            if label_data:
                label_file_path = os.path.join(val_labels_dir, image_file.replace(".png", ".txt").replace(".jpg", ".txt"))
                with open(label_file_path, 'w') as label_file:
                    label_file.write("\n".join(label_data))

def remove_empty_or_nonpair_files():
    # Remove empty label files
    for label_file in os.listdir(train_labels_dir):
        label_path = os.path.join(train_labels_dir, label_file)
        if os.path.getsize(label_path) == 0:
            print(f"Removing empty label file: {label_path}")
            os.remove(label_path)
            image_file = label_file.replace(".txt", ".png").replace(".txt", ".jpg")
            image_path = os.path.join(train_dir, image_file)
            if os.path.exists(image_path):
                os.remove(image_path)
    
    for label_file in os.listdir(val_labels_dir):
        label_path = os.path.join(val_labels_dir, label_file)
        if os.path.getsize(label_path) == 0:
            print(f"Removing empty label file: {label_path}")
            os.remove(label_path)
            image_file = label_file.replace(".txt", ".png").replace(".txt", ".jpg")
            image_path = os.path.join(val_dir, image_file)
            if os.path.exists(image_path):
                os.remove(image_path)

    # Remove non-pair image files
    for image_file in os.listdir(train_dir):
        label_file = image_file.replace(".png", ".txt").replace(".jpg", ".txt")
        if not os.path.exists(os.path.join(train_labels_dir, label_file)):
            print(f"Removing non-pair image file: {image_file}")
            os.remove(os.path.join(train_dir, image_file))

    for image_file in os.listdir(val_dir):
        label_file = image_file.replace(".png", ".txt").replace(".jpg", ".txt")
        if not os.path.exists(os.path.join(val_labels_dir, label_file)):
            print(f"Removing non-pair image file: {image_file}")
            os.remove(os.path.join(val_dir, image_file))

def split_data():
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".png") or f.endswith(".jpg")]
    random.shuffle(image_files)
    
    total_files = len(image_files)
    split_index = int(total_files * 0.8)
    
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    # Move images and labels to respective folders
    for file in train_files:
        shutil.move(os.path.join(images_dir, file), os.path.join(train_dir, file))
        label_file = file.replace(".png", ".txt").replace(".jpg", ".txt")
        shutil.move(os.path.join(labels_dir, "train", label_file), os.path.join(train_labels_dir, label_file))

    for file in val_files:
        shutil.move(os.path.join(images_dir, file), os.path.join(val_dir, file))
        label_file = file.replace(".png", ".txt").replace(".jpg", ".txt")
        shutil.move(os.path.join(labels_dir, "val", label_file), os.path.join(val_labels_dir, label_file))

# Run the script
create_yolo_labels()
remove_empty_or_nonpair_files()
split_data()