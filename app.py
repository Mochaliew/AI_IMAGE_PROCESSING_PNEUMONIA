import streamlit as st
import numpy as np
import cv2
from skimage.feature import hog
import joblib
from tensorflow import keras

IMG_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# Load models
svm_model = joblib.load("svm_model.joblib")
scaler = joblib.load("scaler.joblib")
mobilenet_model = keras.models.load_model("best_model.keras")

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
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

st.title("Pneumonia Detection (HOG + SVM & MobileNetV2)")

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
        mobilenet_pred_proba = mobilenet_model.predict(img_processed, verbose=0)[0][0]
        mobilenet_pred = "PNEUMONIA" if mobilenet_pred_proba > 0.5 else "NORMAL"
        mobilenet_confidence = mobilenet_pred_proba * 100 if mobilenet_pred_proba > 0.5 else (1 - mobilenet_pred_proba) * 100
    
    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("HOG + SVM")
        st.success(f"Prediction: **{svm_pred}**")
    
    with col2:
        st.subheader("MobileNetV2")
        st.success(f"Prediction: **{mobilenet_pred}**")
        st.info(f"Confidence: {mobilenet_confidence:.2f}%")
