
#!/usr/bin/env python3

import cv2
import json
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import numpy as np
import copy

from dataset import Dataset
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('split_dataset/dataset_files.json', 'r') as f:
    dataset_filenames = json.load(f)

test_filenames = dataset_filenames['testing_files'][:500]

test_dataset = Dataset(test_filenames)

with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

labels = list(label_mapping.keys())

batch_size = 24
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def draw_boxes(detections, width, height, frame):

    probabilities = F.softmax(detections, dim=1)

    _, class_indices = torch.max(probabilities, 1)
    
    for i, detection in enumerate(detections):
        print(f"Detection {i + 1}:")

        confidence = probabilities[i, class_indices[i]].item()
        class_id = class_indices[i].item()
        class_name = labels[class_id]
        
        # Extract bounding box coordinates
        box = detection[2:6].detach().cpu().numpy() * np.array([width, height, width, height])
        (x, y, w, h) = box.astype("int")

        print(f"Class_id: {class_id}, Class: {class_name}, Confidence: {confidence:.2f}")
        print(f"Box Coordinates: (x={x}, y={y}, w={w}, h={h})")
        
        if confidence > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    print("\n")

def main():

    model = Model().to(device)

    # Load the trained model
    checkpoint = torch.load('models/checkpoint.pkl', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        height, width, _ = frame.shape

        frame = cv2.flip(frame, 1)

        frame_copy = copy.deepcopy(frame)

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        pil_image = Image.fromarray(frame_copy, mode="RGB")
        input_tensor = transform(pil_image)
        input_data = input_tensor.unsqueeze(0).to(device)

        detections = model.forward(input_data)

        print("Detections:", detections)

        draw_boxes(detections, width, height, frame_copy)

        cv2.imshow('Original Frame', frame)
        cv2.imshow('Objects Detection', frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
