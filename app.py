# app.py
import streamlit as st

# Title of the app
st.title('FB Live Seller Clustering')

# Some text and an input box
model = st.text_input("Select a clustering model")

# If the user inputs their name, show a greeting
model = st.selectbox(
    'Select a clustering model:',
    ('GMM', 'Hierarchical Clustering', 'DBSCAN', 'Self-Organizing Maps')
)

if model == 'GMM' :
    n_components = st.slider(
        'Number of components:', 1, 10, 3
    )

show_diag = st.checkbox('Show Diagram')








