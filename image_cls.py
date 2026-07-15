import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time
import os
import urllib.request

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="🐟 Fish Classification Dashboard",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# CUSTOM CSS - makes it look premium
# ------------------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%);
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .big-title {
        font-size: 42px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #0077b6, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .result-card {
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        padding: 25px;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    .result-card h1 {
        font-size: 30px;
        margin: 0;
    }
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #023047, #0077b6);
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# LOAD MODEL (Downloads from Google Drive if not present locally)
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "densenet_model.keras")

def download_model():
    # Ungaloda Google Drive file ID mathi set seiyapatullathu
    DRIVE_FILE_ID = "1GXdoxKXk4Ak8mlxvqH77zhJ_9E_o3PfY"
    url = f"https://docs.google.com/uc?export=download&id={DRIVE_FILE_ID}"
    
    # User-ku background process theriyavum, confusion thavirkavum loading indicator
    with st.spinner("📥 Model file download aaguthu... (Oru murai mattum, thayavu senju wait pannunga!)"):
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
            st.success("✅ Model successful-a download aaiduchu!")
        except Exception as e:
            st.error(f"❌ Model-ah download panna mudiyala: {e}")
            st.stop()

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        download_model()
    return load_model(MODEL_PATH)

# Intha step-ila model local storage-la illati download panni load aagidum
model = get_model()

# ------------------------------------------------------------------
# CLASS NAMES
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🐟 About this App")
    st.write(
        "Upload a photo of a fish and this app will "
        "tell you what type of fish it is."
    )
    st.markdown("---")
    st.markdown("### 📋 Supported Classes")
    for c in class_names:
        st.markdown(f"- {c}")
    st.markdown("---")
    st.markdown("### ⚙️ How to use")
    st.write("1. Upload a clear fish image\n2. Wait for prediction\n3. View confidence scores below")

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown('<div class="big-title">🐟 Fish Image Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a fish image and instantly find out what type of fish it is</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# FILE UPLOADER
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Choose a fish image...",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------
if uploaded_file is not None:

    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with col2:
        with st.spinner("🔎 Analyzing image..."):
            start = time.time()
            predictions = model.predict(img_array)
            elapsed = time.time() - start

        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index] * 100
        predicted_class = class_names[class_index]

        # Result card
        st.markdown(f"""
            <div class="result-card">
                <h1>🐠 {predicted_class}</h1>
                <p style="font-size:20px; margin-top:8px;">Confidence: <b>{confidence:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)

        st.write("")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
                <div class="metric-box">
                    <h3>⏱ {elapsed:.2f}s</h3>
                    <p>Prediction Time</p>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
                <div class="metric-box">
                    <h3>🎯 {confidence:.1f}%</h3>
                    <p>Top Confidence</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # ALL CLASS PROBABILITIES - Interactive Chart
    # ------------------------------------------------------------------
    st.subheader("🔍 Confidence Score Breakdown")

    df = pd.DataFrame({
        "Fish Class": class_names,
        "Confidence (%)": predictions[0] * 100
    }).sort_values("Confidence (%)", ascending=True)

    fig = px.bar(
        df,
        x="Confidence (%)",
        y="Fish Class",
        orientation="h",
        color="Confidence (%)",
        color_continuous_scale="Blues",
        text=df["Confidence (%)"].apply(lambda x: f"{x:.2f}%")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=450,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 105]),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Expandable raw table
    with st.expander("📊 View as Table"):
        st.dataframe(
            df.sort_values("Confidence (%)", ascending=False).reset_index(drop=True),
            use_container_width=True
        )

else:
    st.info("👆 Upload a fish image above to get started!")
