# Detect-Workplace-Anomalies
We want to build a solution that detects workplace anomalies on live cameras, e.g. fall detection, boundary monitoring, fire, and helmet-wearing.

The first part of the job will be achieving a few of the use cases, and based on the work quality, the rest of the project will be awarded.

Highlevel requirement is below, but there will be more use cases and will provide details to the wining bid.

- Helmet Compliance: Detect when employees are not wearing helmets in designated areas.
- Restricted Area Violations: Detect mobile phone usage, smoking, and loose clothing in hazardous or prohibited zones.
- Fall Detection: Automatically identify and alert on incidents where employees have fallen, ensuring timely responses.
- Boundary Wall Monitoring: Detect breaches or unauthorized access to the facility.
- Unauthorized Movement: Identify movement within restricted areas after working hours.
- Fire and Smoke Detection: Real-time monitoring for the presence of fire or smoke to ensure immediate response.
- Fire Equipment Surveillance: Ensure fire equipment is correctly placed and accessible.
- Employee Monitoring: Detect sleeping during work hours and mishandling of goods.
- Tool and Machine Monitoring: Alert for improper use of equipment and possible pilferage or theft.
- Vehicle and Parking Violations: Monitor vehicle search protocol compliance and parking violations.
- Criminal Activities Detection: Identify scuffles or unauthorized physical altercations in the workplace
- Coal Yard Surveillance: Monitor stacker movement and detect coal smothering or spillage.
- Leakage and Spillage Detection: Identify hazardous material spills or leakages in key areas.
- Height Safety Monitoring: Ensure that workers at height use safety harnesses properly.
- =========================
To build a solution that detects workplace anomalies using live cameras, you will need to utilize computer vision techniques, machine learning models, and real-time video processing. Below is a Python code outline for each of the specified use cases using OpenCV, TensorFlow, and other relevant libraries for detection tasks like helmet compliance, fall detection, and boundary monitoring.
Requirements

    OpenCV for video processing
    TensorFlow/Keras for pre-trained models (e.g., object detection, fall detection, and anomaly detection)
    YOLO (You Only Look Once) for real-time object detection
    Haar Cascades/Deep Learning for facial recognition or helmet detection
    Smoke/fire detection models for fire and smoke detection
    Pre-trained models for anomaly detection or custom-trained models
    OpenCV DNN for deploying deep learning models

Example Solution Breakdown

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load pre-trained models (YOLO for object detection)
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load the Fire Detection Model
fire_model = keras.models.load_model("fire_detection_model.h5")

# Helmet detection using a custom-trained model or predefined classifier
helmet_cascade = cv2.CascadeClassifier("helmet_cascade.xml")

# Initialize Video Capture (Assume camera index is 0)
cap = cv2.VideoCapture(0)

def detect_objects(frame):
    """Detect objects like helmet, phones, unauthorized movements using YOLO"""
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold
                box = detection[0:4] * np.array([width, height, width, height])
                x_center, y_center, w, h = box.astype("int")
                x = int(x_center - w / 2)
                y = int(y_center - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def detect_fall(frame):
    """Detect falls using posture analysis or AI models"""
    # Placeholder for custom fall detection logic
    # You could implement deep learning models for fall detection
    pass

def detect_fire(frame):
    """Detect fire or smoke"""
    frame_resized = cv2.resize(frame, (224, 224))
    fire_pred = fire_model.predict(np.expand_dims(frame_resized, axis=0))
    if fire_pred > 0.5:  # Threshold
        cv2.putText(frame, "Fire Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
def detect_helmet(frame):
    """Detect if helmet is worn"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    helmets = helmet_cascade.detectMultiScale(gray, 1.3, 5)
    if len(helmets) == 0:
        cv2.putText(frame, "Helmet not worn", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def detect_movement(frame):
    """Detect unauthorized movement in restricted areas"""
    # Placeholder for motion detection logic
    pass

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detect_objects(frame)  # Detect objects like helmet or mobile
        detect_fall(frame)      # Detect fall using posture or AI
        detect_fire(frame)      # Fire detection using a custom trained model
        detect_helmet(frame)    # Helmet detection logic

        # Display the frame with annotations
        cv2.imshow('Workplace Anomaly Detection', frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

Explanation:

    Object Detection:
        We use YOLO (You Only Look Once) for object detection to identify helmets, phones, or other items that might violate workplace safety protocols.
    Fall Detection:
        This could be implemented by detecting posture or using a pre-trained fall detection model. Custom models can be built using CNN architectures for this use case.
    Fire and Smoke Detection:
        This leverages a custom fire detection model to identify smoke or fire. You could use TensorFlow or PyTorch to train such a model on fire-related images.
    Helmet Detection:
        We use OpenCV’s Haar Cascade Classifier for detecting helmets. You could create a custom cascade for more accurate detection or fine-tune a deep learning model for this.
    Restricted Area Violations:
        This can be detected by analyzing the movements within certain boundaries. Background subtraction and motion detection techniques can be used to track movement.
    Integration with other systems:
        The system can be extended to include logging violations, sending alerts, and integrating with cloud services for real-time analytics.

Extensions & Deployment:

    You can improve the model training for each specific task using labeled data.
    API integration can allow this system to send real-time alerts (e.g., using Twilio or Firebase).
    Edge devices like Raspberry Pi or NVIDIA Jetson can be used for deployment in industrial settings.

Final Note:

This is a basic implementation to help kick-start the project. For production, consider optimizing performance, ensuring proper data privacy, and working on real-time processing with more advanced models.
