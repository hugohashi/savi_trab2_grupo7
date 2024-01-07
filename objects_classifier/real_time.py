#!/usr/bin/env python3

import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import json
import copy

from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = Model().to(device)
checkpoint = torch.load('models/model.pkl', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Open the video capture (0 for webcam)
cap = cv2.VideoCapture(0)

with open('json_files/label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

labels = list(label_mapping.keys())

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    height, width, _ = frame.shape

    frame = cv2.flip(frame, 1)

    obj_detection = copy.deepcopy(frame)

    # Preprocess the input frame
    input_tensor = transform(obj_detection)
    input_tensor = input_tensor.to(device)
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        detections = model(input_batch)

    # Post-process and draw bounding boxes
    probabilities = F.softmax(detections, dim=1)

    _, class_indices = torch.max(probabilities, 1)
    
    for i, detection in enumerate(detections):
        confidence = probabilities[i, class_indices[i]].item()
        class_id = class_indices[i].item()
        class_name = labels[class_id]
        
        # # Extract bounding box coordinates
        # box = detection[2:6].detach().cpu().numpy() * np.array([width, height, width, height])
        
        # (x, y, w, h) = box.astype("int")
        # (x, y, w, h) = abs(x), abs(y), abs(w), abs(h)

        # print(f"Class_id: {class_id}, Class: {class_name}, Confidence: {confidence:.2f}")
        # print(f"Box Coordinates: (x={x}, y={y}, w={w}, h={h})")
        
        if confidence > 0.8:
        #     cv2.rectangle(frame, (x, y), (x+ w, y + h), (0, 255, 0), 2)
            cv2.putText(obj_detection, f"Object detected: {class_name}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(obj_detection, f"With confidence of: {confidence}", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    # Display the frame with bounding boxes
    cv2.imshow('Original frame', frame)
    cv2.imshow('Object Detection', obj_detection)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
