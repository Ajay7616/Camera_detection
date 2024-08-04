import cv2
import numpy as np
import sqlite3
from datetime import datetime
import os

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create a database to store history
conn = sqlite3.connect('people_count.db')
c = conn.cursor()

# Drop the existing table if it exists and create a new one with the correct schema
c.execute('DROP TABLE IF EXISTS people_count')
c.execute('''CREATE TABLE IF NOT EXISTS people_count
             (id INTEGER PRIMARY KEY, timestamp TEXT, frame_path TEXT, person_count INTEGER)''')
conn.commit()

def store_data(frame_path, person_count):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO people_count (timestamp, frame_path, person_count) VALUES (?, ?, ?)", (timestamp, frame_path, person_count))
    conn.commit()

# Ensure the folder to save images exists
save_folder = 'detected_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    current_person_count = 0
    person_detected = False

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                person_detected = True
                current_person_count += 1

    if current_person_count > 0:
        # Save the frame with the detected person
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_name = f"person_detected_{timestamp}.jpg"
        frame_path = os.path.join(save_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        store_data(frame_path, current_person_count)

    # Display the count of people in the current frame on the video
    cv2.putText(frame, f"People Present: {current_person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Esc key to stop
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
