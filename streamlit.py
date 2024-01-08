import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import nn
import os
import requests

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Define the model architecture - VGG16
def initialize_model(num_classes):
    model = models.vgg16(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, num_classes)])
    model.classifier = nn.Sequential(*features)
    return model

# Function to download the model from Google Drive
def download_model_from_drive(file_id, destination_path):
    URL = "https://drive.google.com/uc?export=download&id=" + file_id
    response = requests.get(URL)
    response.raise_for_status()
    with open(destination_path, "wb") as f:
        f.write(response.content)

# Load the trained model
def load_model():
    model_path = 'vgg_fruit_classifier.pth'
    file_id = '1E3WIT2LpCNepIRMn_UOFaVLxEkTk1V95'

    if not os.path.exists(model_path):
        download_model_from_drive(file_id, model_path)

    model = initialize_model(num_classes=3)  
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image transformation
def transform_image(image):
    transform = data_transforms['train']
    return transform(image).unsqueeze(0) 

# Convert model prediction to readable class
class_names = ['Avocado', 'Orange', 'Pineapple'] 

def prediction_to_class(prediction):
    _, predicted_idx = torch.max(prediction, 1)
    return class_names[predicted_idx.item()]

# Load the model
model = load_model()

# Streamlit user interface
st.set_page_config(page_title='Fruit Classifier', page_icon='fruits.png', layout='centered')
st.title('Fruit Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = transform_image(image)
    with torch.no_grad():
        prediction = model(image)
        st.write(f'Prediction: {prediction_to_class(prediction)}')
