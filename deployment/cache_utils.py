# deployment/cache_utils.py
import streamlit as st

@st.cache_resource
def cached_model_loader(fn, *args, **kwargs):
    return fn(*args, **kwargs)

@st.cache_data
def cached_dataset(loader_fn, *args, **kwargs):
    return loader_fn(*args, **kwargs)
