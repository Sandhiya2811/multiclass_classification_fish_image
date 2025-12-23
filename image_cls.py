import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image


# Load model

MODEL_PATH = "C:/vs_code/project_1/densenet_model.keras"
model = load_model(MODEL_PATH)


# Class names

class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]


st.set_page_config(page_title="🐟 Fish Classification App")
st.title("🐟 Fish Image Classification")
st.write("Upload a fish image and predict the fish category")

uploaded_file = st.file_uploader(
    "Choose a fish image...",
    type=["jpg", "jpeg", "png"]
)


# Prediction

if uploaded_file is not None:

    # Load & show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)

    # Predict (ONLY ONE INPUT)
    predictions = model.predict(img_array)

    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index] * 100
    predicted_class = class_names[class_index]

    # Results
    st.success(f"🐠 Predicted Fish Type: **{predicted_class}**")
    st.info(f"📊 Confidence Score: **{confidence:.2f}%**")

    # All probabilities
    st.subheader("🔍 All Class Confidence Scores")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")
