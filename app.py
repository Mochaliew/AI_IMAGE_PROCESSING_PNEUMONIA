import streamlit as st
import numpy as np
import cv2
from skimage.feature import hog
import joblib
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

IMG_SIZE = (96, 96)
IMG_SIZE_MOBILENET = (224, 224)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

CLASS_NAMES = ['NORMAL', 'BACTERIAL', 'VIRAL']

# Custom CNN Architecture
class ChestXrayCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ChestXrayCNN, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
        
        self.block1 = conv_block(3, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 256)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.fc(x)
        return x

# Load models
svm_model = joblib.load("svm_model.joblib")
scaler = joblib.load("scaler.joblib")
mobilenet_model = keras.models.load_model("best_model.keras")

# Load Custom CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_cnn = ChestXrayCNN(num_classes=3)
custom_cnn.load_state_dict(torch.load("best_chest_xray_model.pth", map_location=device))
custom_cnn.to(device)
custom_cnn.eval()

def extract_hog_features_from_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, IMG_SIZE)
    img_gray = cv2.equalizeHist(img_gray)
    features = hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        block_norm='L2-Hys'
    )
    return features

def preprocess_for_mobilenet(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE_MOBILENET)
    img_preprocessed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    return img_batch

def preprocess_for_custom_cnn(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    return input_batch

st.title("Pneumonia Detection (HOG + SVM & MobileNetV2 & Custom CNN)")

uploaded_file = st.file_uploader("Upload chest X-ray", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", caption="Uploaded X-ray")
    
    with st.spinner("Analyzing..."):
        # HOG + SVM Prediction
        feats = extract_hog_features_from_image(img)
        feats_scaled = scaler.transform([feats])
        svm_pred = svm_model.predict(feats_scaled)[0]
        
        # MobileNetV2 Prediction
        img_processed = preprocess_for_mobilenet(img)
        mobilenet_pred_proba = mobilenet_model.predict(img_processed, verbose=0)[0]
        mobilenet_pred_idx = np.argmax(mobilenet_pred_proba)
        mobilenet_pred = CLASS_NAMES[mobilenet_pred_idx]
        mobilenet_confidence = mobilenet_pred_proba[mobilenet_pred_idx] * 100
        
        # Custom CNN Prediction
        img_processed_cnn = preprocess_for_custom_cnn(img)
        with torch.no_grad():
            output = custom_cnn(img_processed_cnn)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted_idx = torch.max(output, 1)
            cnn_pred = CLASS_NAMES[predicted_idx.item()]
            cnn_confidence = probabilities[predicted_idx].item() * 100
            cnn_probs = probabilities.cpu().numpy()
    
    # Display Results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("HOG + SVM")
        st.success(f"Prediction: **{svm_pred}**")
    
    with col2:
        st.subheader("MobileNetV2")
        st.success(f"Prediction: **{mobilenet_pred}**")
        st.info(f"Confidence: {mobilenet_confidence:.2f}%")
        
        with st.expander("View all probabilities"):
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"{class_name}: {mobilenet_pred_proba[i]*100:.2f}%")
    
    with col3:
        st.subheader("Custom CNN")
        st.success(f"Prediction: **{cnn_pred}**")
        st.info(f"Confidence: {cnn_confidence:.2f}%")
        
        with st.expander("View all probabilities"):
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"{class_name}: {cnn_probs[i]*100:.2f}%")
