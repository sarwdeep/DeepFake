import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

st.markdown(
    """
    <style>
    body {
        background-color: rgb(211, 199, 223);
    }
    </style>
    """,
    unsafe_allow_html=True
)
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    
    # If image has an alpha channel, remove it
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_image(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        prediction_label = "Fake"
    else:
        prediction_label = "Real"
    return prediction_label


model = tf.keras.models.load_model('model.h5')

st.title('Deepfake Image Detection')
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png","webpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Detect Deepfake'):
        prediction_result = predict_image(image, model)
        st.write(f'Prediction: {prediction_result}')