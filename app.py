import streamlit as st
import cv2
import numpy as np
import joblib

IMAGE_SIZE = 128

st.set_page_config(page_title="TraceFinder")

st.title("🔍 TraceFinder – Scanner Identification")

model = joblib.load("model.pkl")

uploaded_file = st.file_uploader(
    "Upload scanned image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_flat = img.flatten().reshape(1, -1)

    prediction = model.predict(img_flat)[0]
    confidence = np.max(model.predict_proba(img_flat)) * 100

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.success(f"🖨 Predicted Scanner: {prediction}")
    st.info(f"📊 Confidence: {confidence:.2f}%")
