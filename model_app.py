import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model("tiny-bolt.h5")

st.title("Tiny Bolt")
st.write("A Mini Image Classifier written in python and is trained on the intel image dataset")

# Set custom Streamlit style using CSS
st.markdown(
    """
    <style>
    .stApp header {
        background-color: #007ACC;
    }
     .img {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 128px;  /* Adjust the height as needed */
        width:128px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


    
if uploaded_image is not None:
    # Preprocess the uploaded image
    image = Image.open(uploaded_image)
    image_array = np.array(image)
    image_array = tf.image.resize(image_array, (128, 128))
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.copy()
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    # Get predictions from the model
    predictions = model.predict(image_array)
    st.markdown("---")  # Add a separator line

    # Get predictions from the model
    predictions = model.predict(image_array)
    label = predictions[0].argmax()
    st.subheader("Predictions:")
    st.write(f'The most likely class is: {label}')