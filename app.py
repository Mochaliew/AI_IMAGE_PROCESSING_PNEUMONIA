import streamlit as st
import numpy as np
import cv2
from skimage.feature import hog
import joblib
from tensorflow import keras
from PIL import Image

# Configuration
IMG_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# Load models
@st.cache_resource
def load_models():
    models = {}
    try:
        models['svm_model'] = joblib.load("svm_model.joblib")
        models['scaler'] = joblib.load("scaler.joblib")
    except:
        st.warning("HOG + SVM model not found")
    
    try:
        models['mobilenet'] = keras.models.load_model("best_model.keras")
    except:
        st.warning("MobileNetV2 model not found")
    
    try:
        models['custom_cnn'] = keras.models.load_model("custom_cnn_model.keras")
    except:
        st.warning("Custom CNN model not found (optional)")
    
    return models

models = load_models()

# Feature extraction for HOG + SVM
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

# Preprocessing for Deep Learning models
def preprocess_for_dl(image, target_size=(224, 224)):
    """Preprocess image for MobileNetV2 and Custom CNN"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

# Prediction functions
def predict_hog_svm(image):
    if 'svm_model' not in models or 'scaler' not in models:
        return None, None
    
    feats = extract_hog_features_from_image(image)
    feats_scaled = models['scaler'].transform([feats])
    pred = models['svm_model'].predict(feats_scaled)[0]
    pred_proba = models['svm_model'].predict_proba(feats_scaled)[0]
    confidence = max(pred_proba) * 100
    
    return pred, confidence

def predict_mobilenet(image):
    if 'mobilenet' not in models:
        return None, None
    
    img_processed = preprocess_for_dl(image, target_size=(224, 224))
    pred_proba = models['mobilenet'].predict(img_processed, verbose=0)[0][0]
    
    # Assuming binary classification: 0 = NORMAL, 1 = PNEUMONIA
    pred = "PNEUMONIA" if pred_proba > 0.5 else "NORMAL"
    confidence = pred_proba * 100 if pred_proba > 0.5 else (1 - pred_proba) * 100
    
    return pred, confidence

def predict_custom_cnn(image):
    if 'custom_cnn' not in models:
        return None, None
    
    img_processed = preprocess_for_dl(image, target_size=(224, 224))
    pred_proba = models['custom_cnn'].predict(img_processed, verbose=0)[0][0]
    
    # Assuming binary classification: 0 = NORMAL, 1 = PNEUMONIA
    pred = "PNEUMONIA" if pred_proba > 0.5 else "NORMAL"
    confidence = pred_proba * 100 if pred_proba > 0.5 else (1 - pred_proba) * 100
    
    return pred, confidence

# Streamlit UI
st.title("🫁 Pneumonia Detection System")
st.markdown("Upload a chest X-ray image to detect pneumonia using multiple models")

# Sidebar for model selection
st.sidebar.header("Model Selection")
use_hog_svm = st.sidebar.checkbox("HOG + SVM", value=True, disabled='svm_model' not in models)
use_mobilenet = st.sidebar.checkbox("MobileNetV2", value=True, disabled='mobilenet' not in models)
use_custom_cnn = st.sidebar.checkbox("Custom CNN", value=True, disabled='custom_cnn' not in models)

# File uploader
uploaded_file = st.file_uploader("Upload chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, channels="BGR", caption="Uploaded X-ray", use_container_width=True)
    
    with col2:
        st.subheader("Predictions")
        
        with st.spinner("Analyzing..."):
            results = []
            
            # HOG + SVM prediction
            if use_hog_svm and 'svm_model' in models:
                pred, conf = predict_hog_svm(img)
                if pred is not None:
                    results.append(("HOG + SVM", pred, conf))
            
            # MobileNetV2 prediction
            if use_mobilenet and 'mobilenet' in models:
                pred, conf = predict_mobilenet(img)
                if pred is not None:
                    results.append(("MobileNetV2", pred, conf))
            
            # Custom CNN prediction
            if use_custom_cnn and 'custom_cnn' in models:
                pred, conf = predict_custom_cnn(img)
                if pred is not None:
                    results.append(("Custom CNN", pred, conf))
        
        # Display results
        if results:
            for model_name, prediction, confidence in results:
                color = "red" if prediction == "PNEUMONIA" else "green"
                st.markdown(f"### {model_name}")
                st.markdown(f"**Prediction:** :{color}[{prediction}]")
                st.progress(confidence / 100)
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                st.divider()
            
            # Ensemble prediction (majority vote)
            if len(results) > 1:
                st.subheader("Ensemble Prediction")
                pneumonia_count = sum(1 for _, pred, _ in results if pred == "PNEUMONIA")
                ensemble_pred = "PNEUMONIA" if pneumonia_count > len(results) / 2 else "NORMAL"
                ensemble_color = "red" if ensemble_pred == "PNEUMONIA" else "green"
                st.markdown(f"### :{ensemble_color}[{ensemble_pred}]")
                st.caption(f"Based on {pneumonia_count}/{len(results)} models predicting PNEUMONIA")
        else:
            st.error("No models available for prediction")

# Information section
with st.expander("ℹ️ About the Models"):
    st.markdown("""
    **HOG + SVM**: Uses Histogram of Oriented Gradients features with Support Vector Machine classifier.
    
    **MobileNetV2**: A lightweight deep learning architecture optimized for mobile and embedded devices.
    
    **Custom CNN**: A custom Convolutional Neural Network designed specifically for chest X-ray classification.
    
    **Note**: For medical diagnosis, always consult with healthcare professionals. This tool is for educational purposes only.
    """)
