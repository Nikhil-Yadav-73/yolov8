import os

def remove_empty_labels(directory):
    for label_file in os.listdir(directory):
        if label_file.endswith('.txt'):
            label_path = os.path.join(directory, label_file)
            try:
                with open(label_path, 'r') as file:
                    content = file.read().strip()
                    if not content:
                        print(f"Removing empty label file: {label_path}")
                        os.remove(label_path)
            except PermissionError as e:
                print(f"PermissionError: Unable to delete {label_path}. Error: {e}")
            except Exception as e:
                print(f"Error with file {label_path}: {e}")

# Remove empty labels from train and val directories
remove_empty_labels(r"C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\license_plates\labels\val")
remove_empty_labels(r"C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\license_plates\labels\train")
