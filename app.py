import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the saved model
model = load_model("flower_classification_model.h5")

# Define image preprocessing function
def preprocess_image(image):
    # Resize image to 128x128
    image_resized = cv2.resize(image, (128, 128))
    # Convert image to RGB (if not already in RGB)
    if len(image_resized.shape) == 2:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    else:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    # Preprocess the image as required by VGG16
    image_preprocessed = preprocess_input(image_resized)
    return image_preprocessed

# Function to make predictions
def predict_class(image):
    preprocessed_image = preprocess_image(image)
    # Expand dimensions to match model input shape
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    # Make prediction
    prediction = model.predict(preprocessed_image)
    # Decode prediction
    class_index = np.argmax(prediction)
    return class_index

# Streamlit app
st.title("Flower Classification Using DEEP LEARNING")

st.write("The model has been trained on a variety of flower types, including Tulip, Rose, Sunflower, Dandelion, and Daisy.")
st.write("For optimal performance, please provide images of these flowers.")


flower_names = {
    0: "DAISY",
    1: "DANDELION",
    2: "ROSE",
    3: "SUNFLOWER",
    4: "TULIP"
}

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    # Display image
    st.image(image, caption='Uploaded Flower Image', width=700, height=300)
    st.write("")
    # Check if predict button is clicked
    if st.button('Predict'):
        # Preprocess and predict
        class_index = predict_class(image)
        predicted_flower = flower_names.get(class_index, "Unknown")
        st.write(f"Predicted Flower: {predicted_flower}")
