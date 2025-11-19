# # streamlit_app.py
# # streamlit_app.py
# import streamlit as st

# st.set_page_config(
#     page_title="ESG FinBERT Dashboard",
#     page_icon="📊",
#     layout="wide"
# )

# st.sidebar.title("streamlit app")

# # Corrected page link
# st.sidebar.page_link("pages/1_📈_Model_Analytics.py", label="Model Analytics")
# st.sidebar.page_link("pages/2_🧠_Live_Inference.py", label="Live Inference")
# st.sidebar.page_link("pages/3_📘_Dataset_Explorer.py", label="Dataset Explorer")
# st.sidebar.page_link("pages/4_⚙️_Model_Info.py", label="Model Info")

# st.write("# Welcome to ESG MultiTask FinBERT Streamlit App")
# st.write("Use the sidebar to navigate between pages.")

import streamlit as st
import os
import sys

# ------------------------------------------------------------
# FIX PATHS — ensure project root is importable
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="ESG FinBERT App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ESG FinBERT Streamlit Application")
st.write("Welcome! Use the sidebar to navigate through the app pages.")
