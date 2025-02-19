import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
model = tf.keras.models.load_model("trained_model.keras")

def preprocess_image(image):
    img_size = 128  # Resize same as training
    img = Image.open(image).convert('L')  # Convert to grayscale
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0  # Normalize
    return img_array.reshape(1, img_size, img_size, 1)

def predict_image(image):
    img_array = preprocess_image(image)
    probability = model.predict(img_array)[0][0]  # Get probability
    confidence = probability * 100 if probability >= 0.5 else (1 - probability) * 100
    label = "NORMAL" if probability >= 0.5 else "PNEUMONIA"
    return label, confidence

# Streamlit UI
st.set_page_config(page_title="Chest X-Ray Classifier", layout="wide")

# Sidebar - Introduction
st.sidebar.title("About the Project")
st.sidebar.write("This is a Deep Learning model to classify **Chest X-rays** into **Normal** or **Pneumonia**.")
st.sidebar.write("**How it works?** Upload an X-ray image, and the model will analyze it to give a prediction.")
st.sidebar.write("### Steps to Use:")
st.sidebar.write("1️⃣ Upload a Chest X-ray image")
st.sidebar.write("2️⃣ Click on 'Analyze' button")
st.sidebar.write("3️⃣ View the prediction with confidence score")

# Main Dashboard
st.title("Chest X-Ray Disease Detection")
st.write("Upload a Chest X-ray image below to classify it as **Normal** or **Pneumonia**.")

uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", width=500)
    with col2:
        st.subheader("Model Prediction")
        prediction, confidence = predict_image(uploaded_file)
        st.success(f"**Prediction: {prediction}**")
        st.info(f"Confidence: {confidence:.2f}%")
