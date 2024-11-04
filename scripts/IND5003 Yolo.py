# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 07:25:37 2024

@author: ansel
"""

from torch import hub # Hub contains other models like FasterRCNN
import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load(
    'ultralytics/yolov5', 
    'yolov5s', 
    pretrained=True
)
# Function to apply Gaussian Blur
def apply_gaussian_blur(img, kernel_size=(5, 5)):
    return cv2.GaussianBlur(img, kernel_size, 0)

# Function to apply Median Filtering
def apply_median_filter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)

# Function to apply Non-Local Means Denoising
def apply_nlm_denoising(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# Function to apply Bilateral Filtering
def apply_bilateral_filter(img, diameter=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)

# Define a function to plot the boxes on the frame
def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    
    for i in range(n):
        row = cord[i]
        # If confidence score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.2: 
            continue
        
        # Convert normalized coordinates to image coordinates
        x1 = int(row[0] * x_shape)
        y1 = int(row[1] * y_shape)
        x2 = int(row[2] * x_shape)
        y2 = int(row[3] * y_shape)
        
        # Set the color of the bounding box and label text
        bgr = (0, 255, 0)  # Green for bounding box
        label_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for label

        # Get class names from the model
        classes = model.names

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

        # Put a label above the bounding box
        cv2.putText(frame,
                    f"{classes[labels[i]]} {row[4]:.2f}",  # Class name and confidence
                    (x1, y1 - 10),
                    label_font, 0.9, bgr, 2)

    return frame

# Function to perform object detection on an image
# Function to perform object detection on an image with enhancements
def detect_on_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    

    # Convert image to RGB (OpenCV uses BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform inference using the YOLOv5 model
    results = model(img_rgb)

    # Extract the labels and bounding box coordinates
    labels = results.xyxyn[0][:, -1].cpu().numpy()  # Class labels
    cord = results.xyxyn[0][:, :-1].cpu().numpy()  # Bounding box coordinates and confidence scores

    # Plot the bounding boxes on the image
    output_img = plot_boxes((labels, cord), img)

    # Display the image with detections
    cv2.imshow("Detected Image", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# Example usage:
image_path = '6701_0618_20241012062015_101100.jpg'  # Replace with your image path
detect_on_image(image_path)

