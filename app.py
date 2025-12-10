import streamlit as st
import numpy as np
import cv2
from skimage.feature import hog
import joblib

IMG_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# Load model + scaler
svm_model = joblib.load("svm_model_fast.pkl")
scaler = joblib.load("scaler_fast.pkl")

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

st.title("Pneumonia Detection (HOG + SVM)")

uploaded_file = st.file_uploader("Upload chest X-ray", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Uploaded X-ray")

    with st.spinner("Analyzing..."):
        feats = extract_hog_features_from_image(img)
        feats_scaled = scaler.transform([feats])
        pred = svm_model.predict(feats_scaled)[0]

    st.success(f"Prediction: **{pred}**")
