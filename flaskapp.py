from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

app = Flask(__name__)

class_names = {
    0:"dry",
    1: "hazardous",
    2: "recyclable",
    3: "wet"
}
# Load your model
model = torch.load('/home/sevengods/Documents/WMS&LD/Litter_detection_model.pth', map_location=torch.device('cpu'))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def home():
    return app.send_static_file('wastewebsite.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the request
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    # Apply the transformation
    img = transform(img).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        output = model(img)
    predicted_label = int(np.argmax(output.numpy()))

     # Get the class name for the predicted label
    predicted_class_name = class_names[predicted_label]

    # Return the predicted label
    return jsonify({'predicted_label': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
