import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from datetime import datetime
import csv
import os

# -------------------- SELECT MODE --------------------
print("Select Mode:")
print("1 : Upload Image")
print("2 : Use Camera")

mode = input("Enter option (1 or 2): ")

# -------------------- LOAD MODELS --------------------
yolo = YOLO("ml-attandes_system/yolov8n.pt")
model = load_model(" # enter your trained model")

print("Models loaded successfully")

# -------------------- CLASS NAMES --------------------
class_names = ["Aamir Khan images", "RaAbhishek Bachchan imageshul", "Aditya Roy Kapur images","Aftab Shivdasani images","Ajay Devgn images","Akshay Kumar images","Amitabh Bachchan images","Anil Kapoor images","Arjun Kapoor images","Bobby Deol images1"]

# -------------------- ATTENDANCE --------------------
attendance_file = "attendance.csv"
marked_names = set()

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Time"])

def mark_attendance(name):
    if name not in marked_names:
        with open(attendance_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            )
        marked_names.add(name)
        print(f"Attendance marked for {name}")

# -------------------- FACE PROCESS FUNCTION --------------------
def process_frame(frame):
    results = yolo(frame, conf=0.5)

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            preds = model.predict(face, verbose=0)
            idx = np.argmax(preds)
            confidence = preds[0][idx]

            if confidence > 0.85:
                name = class_names[idx]
                mark_attendance(name)
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{name} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)
    return frame

# -------------------- IMAGE MODE --------------------
if mode == "1":
    img_path = input("Enter image path: ")

    if not os.path.exists(img_path):
        print("❌ Image not found")
        exit()

    image = cv2.imread(img_path)
    image = process_frame(image)

    cv2.imshow("Image Attendance", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------- CAMERA MODE --------------------
elif mode == "2":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not opened")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        cv2.imshow("Camera Attendance", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid option")
