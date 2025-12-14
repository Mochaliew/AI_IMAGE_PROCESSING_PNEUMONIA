import streamlit as st
import numpy as np
import cv2
from skimage.feature import hog
import joblib
from tensorflow import keras
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ============ HOG PARAMETERS ============
IMG_SIZE = (96, 96)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# ============ CLASS NAMES ============
CLASS_NAMES = ['NORMAL', 'BACTERIAL', 'VIRAL']

# ============ CUSTOM CNN ARCHITECTURE ============
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

# ============ LOAD MODELS ============
@st.cache_resource
def load_all_models():
    """Load all three models and cache them"""
    models = {}
    errors = []
    
    # Load SVM Model
    try:
        models['svm'] = joblib.load("svm_model.joblib")
        models['scaler'] = joblib.load("scaler.joblib")
        st.sidebar.success("✅ SVM model loaded")
    except Exception as e:
        errors.append(f"SVM: {str(e)}")
        models['svm'] = None
    
    # Load MobileNetV2 Model
    try:
        models['mobilenet'] = keras.models.load_model("best_model.keras")
        st.sidebar.success("✅ MobileNetV2 model loaded")
    except Exception as e:
        errors.append(f"MobileNetV2: {str(e)}")
        models['mobilenet'] = None
    
    # Load Custom CNN Model (PyTorch)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_model = ChestXrayCNN(num_classes=3)
        cnn_model.load_state_dict(torch.load("best_chest_xray_model.pth", map_location=device))
        cnn_model.to(device)
        cnn_model.eval()
        models['custom_cnn'] = cnn_model
        models['device'] = device
        st.sidebar.success(f"✅ Custom CNN loaded ({device})")
    except Exception as e:
        errors.append(f"Custom CNN: {str(e)}")
        models['custom_cnn'] = None
    
    if errors:
        st.sidebar.warning("⚠️ Some models failed to load:")
        for error in errors:
            st.sidebar.text(error)
    
    return models

# Load all models
models = load_all_models()

# ============ PREPROCESSING FUNCTIONS ============
def extract_hog_features_from_image(image):
    """Extract HOG features for SVM"""
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
    """Preprocess image for MobileNetV2 (TensorFlow)"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

def preprocess_for_custom_cnn(image):
    """Preprocess image for Custom CNN (PyTorch)"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    
    # PyTorch transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(models.get('device', 'cpu'))
    return input_batch

# ============ STREAMLIT UI ============
st.title("🫁 Pneumonia Detection - 3 Model Comparison")
st.write("Compare predictions from HOG+SVM, MobileNetV2, and Custom CNN")

# Show model status
with st.expander("📊 Model Status"):
    col1, col2, col3 = st.columns(3)
    with col1:
        if models.get('svm'):
            st.success("✅ HOG + SVM")
        else:
            st.error("❌ HOG + SVM")
    with col2:
        if models.get('mobilenet'):
            st.success("✅ MobileNetV2")
        else:
            st.error("❌ MobileNetV2")
    with col3:
        if models.get('custom_cnn'):
            st.success("✅ Custom CNN")
        else:
            st.error("❌ Custom CNN")

# File uploader
uploaded_file = st.file_uploader("Upload chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Display uploaded image
    st.image(img, channels="BGR", caption="Uploaded X-ray")
    
    with st.spinner("Analyzing with all models..."):
        results = {}
        
        # ========== HOG + SVM PREDICTION ==========
        if models.get('svm') and models.get('scaler'):
            try:
                feats = extract_hog_features_from_image(img)
                feats_scaled = models['scaler'].transform([feats])
                svm_pred = models['svm'].predict(feats_scaled)[0]
                results['svm'] = {
                    'prediction': svm_pred,
                    'success': True
                }
            except Exception as e:
                results['svm'] = {'success': False, 'error': str(e)}
        else:
            results['svm'] = {'success': False, 'error': 'Model not loaded'}
        
        # ========== MOBILENETV2 PREDICTION ==========
        if models.get('mobilenet'):
            try:
                img_processed = preprocess_for_mobilenet(img)
                mobilenet_pred_proba = models['mobilenet'].predict(img_processed, verbose=0)[0]
                mobilenet_pred_idx = np.argmax(mobilenet_pred_proba)
                mobilenet_pred = CLASS_NAMES[mobilenet_pred_idx]
                mobilenet_confidence = mobilenet_pred_proba[mobilenet_pred_idx] * 100
                
                results['mobilenet'] = {
                    'prediction': mobilenet_pred,
                    'confidence': mobilenet_confidence,
                    'probabilities': mobilenet_pred_proba,
                    'success': True
                }
            except Exception as e:
                results['mobilenet'] = {'success': False, 'error': str(e)}
        else:
            results['mobilenet'] = {'success': False, 'error': 'Model not loaded'}
        
        # ========== CUSTOM CNN PREDICTION ==========
        if models.get('custom_cnn'):
            try:
                img_processed = preprocess_for_custom_cnn(img)
                with torch.no_grad():
                    output = models['custom_cnn'](img_processed)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    _, predicted_idx = torch.max(output, 1)
                    
                    cnn_pred = CLASS_NAMES[predicted_idx.item()]
                    cnn_confidence = probabilities[predicted_idx].item() * 100
                    cnn_probs = probabilities.cpu().numpy()
                
                results['custom_cnn'] = {
                    'prediction': cnn_pred,
                    'confidence': cnn_confidence,
                    'probabilities': cnn_probs,
                    'success': True
                }
            except Exception as e:
                results['custom_cnn'] = {'success': False, 'error': str(e)}
        else:
            results['custom_cnn'] = {'success': False, 'error': 'Model not loaded'}
    
    # ========== DISPLAY RESULTS ==========
    st.success("✅ Analysis complete!")
    
    col1, col2, col3 = st.columns(3)
    
    # HOG + SVM Results
    with col1:
        st.subheader("🔬 HOG + SVM")
        if results['svm']['success']:
            pred = results['svm']['prediction']
            if pred == "NORMAL":
                st.success(f"**{pred}**")
            else:
                st.error(f"**{pred}**")
        else:
            st.error(f"❌ Error: {results['svm']['error']}")
    
    # MobileNetV2 Results
    with col2:
        st.subheader("🤖 MobileNetV2")
        if results['mobilenet']['success']:
            pred = results['mobilenet']['prediction']
            conf = results['mobilenet']['confidence']
            if pred == "NORMAL":
                st.success(f"**{pred}**")
            else:
                st.error(f"**{pred}**")
            st.info(f"Confidence: {conf:.2f}%")
            
            with st.expander("View probabilities"):
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = results['mobilenet']['probabilities'][i] * 100
                    st.progress(prob / 100, text=f"{class_name}: {prob:.2f}%")
        else:
            st.error(f"❌ Error: {results['mobilenet']['error']}")
    
    # Custom CNN Results
    with col3:
        st.subheader("🧠 Custom CNN")
        if results['custom_cnn']['success']:
            pred = results['custom_cnn']['prediction']
            conf = results['custom_cnn']['confidence']
            if pred == "NORMAL":
                st.success(f"**{pred}**")
            else:
                st.error(f"**{pred}**")
            st.info(f"Confidence: {conf:.2f}%")
            
            with st.expander("View probabilities"):
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = results['custom_cnn']['probabilities'][i] * 100
                    st.progress(prob / 100, text=f"{class_name}: {prob:.2f}%")
        else:
            st.error(f"❌ Error: {results['custom_cnn']['error']}")
    
    # Consensus Summary
    st.divider()
    st.subheader("📊 Model Consensus")
    
    successful_predictions = []
    if results['svm']['success']:
        successful_predictions.append(results['svm']['prediction'])
    if results['mobilenet']['success']:
        successful_predictions.append(results['mobilenet']['prediction'])
    if results['custom_cnn']['success']:
        successful_predictions.append(results['custom_cnn']['prediction'])
    
    if successful_predictions:
        from collections import Counter
        vote_counts = Counter(successful_predictions)
        most_common = vote_counts.most_common(1)[0]
        
        if most_common[1] >= 2:  # At least 2 models agree
            st.success(f"🎯 **Consensus: {most_common[0]}** ({most_common[1]}/{len(successful_predictions)} models agree)")
        else:
            st.warning("⚠️ Models disagree - review individual predictions carefully")
            for pred, count in vote_counts.items():
                st.write(f"- {pred}: {count} model(s)")
    else:
        st.error("❌ No successful predictions")

else:
    st.info("👆 Please upload a chest X-ray image to begin analysis")
    
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        ### Three Different Approaches:
        
        1. **HOG + SVM** 
           - Classical machine learning approach
           - Extracts hand-crafted features (Histogram of Oriented Gradients)
           - Fast and lightweight
        
        2. **MobileNetV2** 
           - Transfer learning with pre-trained CNN
           - Based on ImageNet weights
           - Efficient mobile-optimized architecture
        
        3. **Custom CNN** 
           - Custom PyTorch architecture
           - 4 convolutional blocks
           - Trained from scratch on chest X-rays
        
        **Consensus prediction** combines all three models for more reliable results.
        """)
