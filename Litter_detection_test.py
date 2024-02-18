import os
import random
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision

class_names = {
    0:"dry",
    1: "hazardous",
    2: "recyclable",
    3: "wet"
}

# Load your pretrained model
model=torch.load('/home/sevengods/Documents/WMS&LD/Litter_detection_model.pth', map_location=torch.device('cpu'))
model.eval()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path, model, transform):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        return predicted_class, probabilities[predicted_class].item()
    
# Path to the folder containing test images
folder_path = '/home/sevengods/Documents/WMS&LD/roboflow/test'

# List all files in the folder
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Randomly select an image path
random_image_path = random.choice(image_paths)

# Predict the label for the randomly selected image
predicted_class, confidence = predict_image(random_image_path, model, transform)
label = class_names[predicted_class]  # Assuming you have a list of class names

# Display the randomly selected image along with its predicted label and confidence score
image = Image.open(random_image_path)
plt.imshow(np.array(image))
plt.axis('off')
plt.title(f'Predicted Label: {label} (Confidence: {confidence:.2f})')
plt.show()
