# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 09:21:14 2024

@author: ansel
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to perform object detection
def detect_cars(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    img_tensor = torchvision.transforms.ToTensor()(img_rgb)
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Process outputs
    boxes = outputs[0]['boxes'].numpy()
    scores = outputs[0]['scores'].numpy()
    labels = outputs[0]['labels'].numpy()

    # Filter results for cars (COCO class ID for car is 3)
    car_boxes = boxes[labels == 3]
    car_scores = scores[labels == 3]

    # Plotting the results
    for i in range(len(car_boxes)):
        if car_scores[i] > 0.5:  # Set a confidence threshold
            x1, y1, x2, y2 = car_boxes[i]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'Car: {car_scores[i]:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Detected Cars", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = '6701_0618_20241012062015_101100.jpg'  # Replace with your image path
detect_cars(image_path)
