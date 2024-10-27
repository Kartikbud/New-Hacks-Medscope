import warnings
warnings.filterwarnings('ignore', category=Warning)
import os
import urllib.request

import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import google.generativeai as genai
from torchvision.models import ResNet50_Weights
import numpy as np

def download_imagenet_classes():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    filename = "imagenet_classes.txt"
    if not os.path.exists(filename):
        print (f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete")

download_imagenet_classes()

try:
    with open('imagenet_classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(class_names)} classes")
except FileNotFoundError:
    print("Error: imagenet_classes.txt not found!")
    print("Please download it from: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    exit(1)
except Exception as e:
    print(f"Error loading class names: {str(e)}")
    exit(1)

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

def detect_objects(image_path):
    try:
        if not class_names:
            raise ValueError("Class names not loaded properly")
            
        # Use the model's preprocessing transform
        preprocess = ResNet50_Weights.DEFAULT.transforms()
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_processed = preprocess(img).unsqueeze(0)
        
        # Get model predictions
        with torch.no_grad():
            output = model(img_processed)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top 5 predictions safely
        num_classes = min(5, len(class_names))
        top5_prob, top5_indices = torch.topk(probabilities, num_classes)
        
        results = []
        for i in range(num_classes):
            idx = top5_indices[i].item()
            if idx < len(class_names):  # Safety check
                prob = top5_prob[i].item() * 100
                if prob > 1:  # Only include if confidence > 1%
                    results.append((class_names[idx], prob))
        
        return results
        
    except Exception as e:
        print(f"Error in detect_objects: {str(e)}")
        # Return empty list instead of failing
        return []

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    print("Press SPACE to capture an image or ESC to quit")
    while True:
        ret, frame = cap.read()
        if ret:
            # Improve image quality
            frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)  # Increase contrast and brightness
            
        cv2.imshow('Capture Image', frame)
        
        k = cv2.waitKey(1)
        if k%256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:  # SPACE pressed
            # Save image with better quality
            cv2.imwrite('captured_image.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print("Image captured!")
            break

    cap.release()
    cv2.destroyAllWindows()
    return 'captured_image.jpg'

def ask_gemini(objects):
    genai.configure(api_key='AIzaSyCoehh5I1aVSTTeCbFv5zKlgm3lZU2iDRY')
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"Given the following objects: {', '.join([obj[0] for obj in objects])}, what emergency medical tools can be built using these objects? Please provide detailed instructions for 5 possible tools."

    response = model.generate_content(prompt)
    return response.text

def main():
    image_path = capture_image()
    detected_objects = detect_objects(image_path)
    
    if not detected_objects:
        print("No objects detected with high confidence")
        print("Try the following:")
        print("1. Ensure the object is well lit")
        print("2. Center the object in the frame")
        print("3. Move closer to the object")
        print("4. Reduce motion blur by holding the camera steady")
        return
    
    print("Detected objects:")
    for obj, score in detected_objects:
        print(f"{obj}: {score:.2f}%")
    
    emergency_tools = ask_gemini(detected_objects)
    print("\nEmergency medical tools that can be built:")
    print(emergency_tools)

if __name__ == "__main__":
    main()