import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

st.header('Flower Classification CNN Model')

flower_names = ['daisy', 'sunflower', 'tulip']

try:
    model = load_model('PhanLoaiHoa.keras')
except Exception as e:
    st.error(f"Không thể tải mô hình: {e}")
    st.stop()

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_label = np.argmax(result)
    predicted_flower = flower_names[predicted_label]
    outcome = f"Bức ảnh này là {predicted_flower} với độ chính xác {np.max(result)*100:.2f}%"
    return outcome

uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    save_path = os.path.join('CnnProject/upload/', uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width=200)

    st.markdown(classify_images(save_path))