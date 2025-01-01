import cv2
import torch
from ultralytics import YOLO
import pytesseract
import re
from collections import defaultdict


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO(r'/home/nikhil/Desktop/numplate/yolov8/runs/detect/train16/weights/best.pt')

video_file_path = r"C:\Users\Nikhil\Downloads\WhatsApp Video 2024-12-13 at 9.05.25 PM.mp4"

# Open the video file
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Set up the window name
window_name = "YOLOv8 Real-Time Number Plate Detection"

# Define the valid state codes
VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA", "GJ", "HR", "HP",
    "JK", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OR", "PY", "PN",
    "RJ", "SK", "TN", "TR", "UP", "WB"
}

# Initialize the 10-character string
current_plate = list("AA00AA0000")

# Initialize a frequency tracker for each character position
frequency_tracker = [defaultdict(int) for _ in range(10)]

def extract_valid_plate(text):
    # Normalize the text to uppercase and remove whitespace
    text = text.upper().replace(" ", "")
    # Find all substrings matching the AA00AA0000 pattern
    matches = re.findall(r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}', text)
    
    # Handle cases where the text is not exactly matching the expected format (AA00AA0000)
    if matches:
        # Return the first valid match
        return matches[0]
    return None

def update_plate_tracker(plate, tracker):
    """
    Updates the frequency tracker for the given plate.
    """
    for i, char in enumerate(plate):
        tracker[i][char] += 1

def get_most_frequent_plate(tracker):
    """
    Constructs the most frequent plate based on the frequency tracker.
    """
    most_frequent_plate = []
    for char_freq in tracker:
        if char_freq:
            most_frequent_plate.append(max(char_freq, key=char_freq.get))
        else:
            most_frequent_plate.append("0")  # Default to '0' if no data
    return "".join(most_frequent_plate)

while True:
    # Read a frame from the video file
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to read frame from video file.")
        break

    # Perform inference on the frame
    results = model(frame, stream=True)  # Stream mode for real-time processing

    # Flag to track if any license plate was detected
    detected = False

    # Iterate through detections and draw boxes
    for result in results:
        for box in result.boxes:
            # Extract box coordinates, confidence, and class
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box
            confidence = box.conf[0].item()            # Confidence score
            cls = int(box.cls[0].item())               # Class ID

            # Filter detections for license plates only (adjust class ID based on your model training)
            if cls == 0:  # Assuming class '0' is for license plates
                # Crop the detected license plate region
                license_plate = frame[int(y1):int(y2), int(x1):int(x2)]

                # Preprocess the cropped license plate
                license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                license_plate_thresh = cv2.threshold(license_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # Perform OCR on the preprocessed license plate
                plate_text = pytesseract.image_to_string(license_plate_thresh, config='--psm 7')  # psm 7 optimizes for single text lines

                # Clean and extract valid license plate (now we print any OCR result)
                ocr_result = plate_text.strip().replace(" ", "")
                print("OCR Result for Current Frame:", ocr_result)

                valid_plate = extract_valid_plate(ocr_result)

                if valid_plate:
                    # Update the frequency tracker
                    update_plate_tracker(valid_plate, frequency_tracker)

                    # Update the current plate
                    current_plate = list(get_most_frequent_plate(frequency_tracker))

                    # Display the OCR result on the frame
                    label = f"{valid_plate} ({confidence:.2f})"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Set detected flag to True
                    detected = True

    # Always print the current most frequent plate every frame
    print("Current Most Frequent Plate:", "".join(current_plate))

    # Display the frame with detections
    cv2.imshow(window_name, frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Output the final most frequent plate
final_plate = "".join(get_most_frequent_plate(frequency_tracker))
print("Final Most Frequent License Plate:", final_plate)