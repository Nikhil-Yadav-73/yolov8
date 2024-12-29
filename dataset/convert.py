import os
import xml.etree.ElementTree as ET

# Define paths
xml_folder = "./xml_annotation/"
output_folder = "./license_plates/labels/"
images_folder = "./license_plates/images/"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define the class names (adjust if necessary)
classes = ["license_plate"]

def convert_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(images_folder, image_name)
    
    if not os.path.exists(image_path):
        print(f"Image {image_name} not found, skipping...")
        return

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    yolo_annotations = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in classes:
            continue

        class_id = classes.index(class_name)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Normalize coordinates
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        bbox_width = (xmax - xmin) / img_width
        bbox_height = (ymax - ymin) / img_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

    # Save to YOLO format
    txt_file = os.path.join(output_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_annotations))

# Process all XML files
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        convert_to_yolo(os.path.join(xml_folder, xml_file))

print("Conversion completed!")
