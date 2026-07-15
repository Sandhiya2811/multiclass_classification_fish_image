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
class_names =
