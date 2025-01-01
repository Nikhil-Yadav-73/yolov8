import cv2
import torch
from ultralytics import YOLO
import pytesseract
import re
from collections import defaultdict


pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

model = YOLO(r'/home/nikhil/Desktop/numplate/yolov8/runs/detect/train16/weights/best.pt')

video_file_path = r"/home/nikhil/Desktop/numplate/yolov8/vid1.mp4"

cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

window_name = "YOLOv8 Real-Time Number Plate Detection"

VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA", "GJ", "HR", "HP",
    "JK", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OR", "PY", "PN",
    "RJ", "SK", "TN", "TR", "UP", "WB"
}

current_plate = list("AA00AA0000")

frequency_tracker = [defaultdict(int) for _ in range(10)]

def extract_valid_plate(text):
    text = text.upper().replace(" ", "")
    matches = re.findall(r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}', text)
    
    if matches:
        return matches[0]
    return None

def update_plate_tracker(plate, tracker):
    for i, char in enumerate(plate):
        tracker[i][char] += 1

def get_most_frequent_plate(tracker):

    most_frequent_plate = []
    for char_freq in tracker:
        if char_freq:
            most_frequent_plate.append(max(char_freq, key=char_freq.get))
        else:
            most_frequent_plate.append("0")
    return "".join(most_frequent_plate)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to read frame from video file.")
        break

    results = model(frame, stream=True)

    detected = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  
            confidence = box.conf[0].item()            
            cls = int(box.cls[0].item())  

            if cls == 0:
                license_plate = frame[int(y1):int(y2), int(x1):int(x2)]

                license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                license_plate_thresh = cv2.threshold(license_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                plate_text = pytesseract.image_to_string(license_plate_thresh, config='--psm 7')

                ocr_result = plate_text.strip().replace(" ", "")
                print("OCR Result for Current Frame:", ocr_result)

                valid_plate = extract_valid_plate(ocr_result)

                if valid_plate:
                    update_plate_tracker(valid_plate, frequency_tracker)

                    current_plate = list(get_most_frequent_plate(frequency_tracker))

                    label = f"{valid_plate} ({confidence:.2f})"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    detected = True

    print("Current Most Frequent Plate:", "".join(current_plate))

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

final_plate = "".join(get_most_frequent_plate(frequency_tracker))
print("Final Most Frequent License Plate:", final_plate)