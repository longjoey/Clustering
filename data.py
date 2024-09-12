import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved.pkl', 'rb') as file:
     data = pickle.load(file)
    
    return data

data = load_model()

model = data["model"]

def show_predict_page():
    st.title("FB Live Seller Clustering")
    
    st.write("""### FB Live Seller Clustering""")
