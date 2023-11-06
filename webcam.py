import math 
import cv2
from ultralytics import YOLO

# start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(1920)
# cap.set(1080)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# model
model = YOLO('runs/detect/plant_detection_1280-250-32_yolov8n4/weights/best.pt')

classNames = ['Cassava Bacterial Blight', 'Cassava Brown Leaf Spot', 'Cassava Healthy', 'Cassava Mosaic', 'Cassava Root Rot', 'Corn Brown Spots', 'Corn Charcoal', 'Corn Chlorotic Leaf Spot', 'Corn Gray leaf spot', 'Corn Healthy', 'Corn Insects Damages', 'Corn Mildew', 'Corn Purple Discoloration', 'Corn Smut', 'Corn Streak', 'Corn Stripe', 'Corn Violet Decoloration', 'Corn Yellow Spots', 'Corn Yellowing', 'Corn leaf blight', 'Corn rust leaf', 'Tomato Brown Spots', 'Tomato bacterial wilt', 'Tomato blight leaf', 'Tomato healthy', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus']

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()