import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader 
from torchinfo import summary
from torchvision import models
from PIL import Image

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to a fixed size
    transforms.ToTensor(),         # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Load data folders
train_data_folder=torchvision.datasets.ImageFolder(root="/home/sevengods/Documents/WMS&LD/roboflow/train",transform=transform)
test_data_folder=torchvision.datasets.ImageFolder(root="/home/sevengods/Documents/WMS&LD/roboflow/valid",transform=transform)
# Split to training and testing sets
# total_samples=len(train_data_folder) # Calculate total number osf samples

'''# Define indices for training and testing
train_size=int(0.8*total_samples)
test_size=total_samples-train_size
train_indices = list(range(train_size))
test_indices = list(range(train_size, total_samples))'''

# Create dataloaders for training and testing
batch_size=8
train_loader = DataLoader(dataset=train_data_folder, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data_folder, batch_size=batch_size, shuffle=True)
class_names=train_data_folder.classes
# print(class_names)
# print(train_loader)
# print(test_loader)

# Pretrained model using resnet101
weights=torchvision.models.ResNet50_Weights.DEFAULT # Default gives the best available weights
model=torchvision.models.resnet50(weights=weights).to(device)
# Freeze all layers except the final classification layer
for name, param in model.named_parameters():
    if "fc" not in name:  # 'fc' is the name of the final classification layer
        param.requires_grad = False

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)

# Print a summary using torchinfo (uncomment for actual output)
'''summary(model=model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
'''

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs=3

# Training loop
n_total_steps=len(train_loader)
for epoch in range(epochs):
    model.train()  # Set the model in training mode
    print(f'Epoch [{epoch + 1}/{epochs}]')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Zero the gradient buffers
        output = model(data)  # Forward pass
        loss = loss_fn(output, target)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights
        if (batch_idx+1) %40==0:
             print(f'Epoch[{epoch+1}/{epochs}],Step[{batch_idx+1}/{n_total_steps}],Loss:{loss.item():.4f}')
    #print(data)
    #print(target)
    #print(batch_idx)

'''# Validation loop (optional)
    model.eval()  # Set the model in evaluation mode
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)'''
train_class_labels=set(class_names)
test_class_labels=set()
with torch.no_grad():
    n_correct=0
    n_samples=0
    n_class_correct=[0 for i in class_names]
    n_class_samples=[0 for i in class_names]
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        test_class_labels.update(labels)
        # max returns (value,index)
        _, predicted=torch.max(outputs,1)
        n_samples+=labels.size(0)
        n_correct+=(predicted==labels).sum().item()

        for i in range(len(labels)):
            label=labels[i].item()
            pred=predicted[i].item()
            if (label==pred):
                n_class_correct[label]+=1
            n_class_samples[label]+=1
    #print(n_correct)
    #print(n_samples)
    #print(n_class_correct)
    #print(n_class_samples)
    acc=100.0*n_correct/n_samples
    #print(f'Accuracy of the network:{acc}%')

    for i,class_name in enumerate(class_names):
        acc=100.0*n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of {class_name}:{acc}%')    

model_path="Litter_detection_model.pth"
#Save the model and associated information
torch.save(model,model_path)