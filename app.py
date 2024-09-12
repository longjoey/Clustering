# app.py
import streamlit as st

# Title of the app
st.title('My Streamlit App')

# Some text and an input box
name = st.text_input("What's your name?")

# If the user inputs their name, show a greeting
if name:
    st.write(f"Hello, {name}!")
