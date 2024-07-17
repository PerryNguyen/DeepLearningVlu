import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'sunflower', 'tulip']

model = load_model('PhanLoaiHoa.keras')
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size = (180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = "Bức ảnh này là " + flower_names[np.argmax(result)] + " với độ chính xác " + str(np.max(result)*100)
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('CnnProject/upload/', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width = 200)

st.markdown(classify_images(uploaded_file))
