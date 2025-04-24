import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cnn_cifar10_model.h5')
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


# Preprocess the image
def preprocess_image(image):
    image = ImageOps.fit(image, (32, 32), Image.LANCZOS)  # Updated here
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Streamlit app
st.title("CIFAR-10 Image Classification")
st.header("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    st.write("Classifying...")
    image = preprocess_image(image)
    prediction = model.predict(image)
    class_name = class_names[np.argmax(prediction)]

    st.markdown(f"<h2 style='text-align: center; color: white;'>Prediction: {class_name}</h2>", unsafe_allow_html=True)