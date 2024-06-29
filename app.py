import streamlit as st
import requests
from PIL import Image
import io
import os

st.title("Melanoma Detection App")

# Endpoint de formation du modèle
if st.button("Train Model"):
    training_dir = 'training/'
    filenames = os.listdir(training_dir)

    data = {
        "filenames": filenames,
        "labels": [1 if f.startswith('melanoma_') else 0 for f in filenames]
    }

    response = requests.post("http://localhost:8000/training", json=data)
    st.write(response.json())

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
    response = requests.post("http://localhost:8000/predict", files=files)
    st.write(response.json())
