# streamlit_app.py
# streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="ESG FinBERT Dashboard",
    page_icon="📊",
    layout="wide"
)

st.sidebar.title("streamlit app")

# Corrected page link
st.sidebar.page_link("pages/1_📈_Model_Analytics.py", label="Model Analytics")
st.sidebar.page_link("pages/2_🧠_Live_Inference.py", label="Live Inference")
st.sidebar.page_link("pages/3_📘_Dataset_Explorer.py", label="Dataset Explorer")
st.sidebar.page_link("pages/4_⚙️_Model_Info.py", label="Model Info")

st.write("# Welcome to ESG MultiTask FinBERT Streamlit App")
st.write("Use the sidebar to navigate between pages.")