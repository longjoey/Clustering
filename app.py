# app.py
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import pandas as pd


# Title of the app
st.title('FB Live Seller Clustering')

url = 'https://raw.githubusercontent.com/longjoey/Clustering/main/X_pca.csv'

def load_data():
    return pd.read_csv(url)

X_pca = load_data()

# If the user inputs their name, show a greeting
model = st.selectbox(
    'Select a clustering model:',
    ('GMM', 'Hierarchical Clustering', 'DBSCAN', 'Self-Organizing Maps')
)

if model == 'GMM' :
    n_components = st.slider(
        'Number of components:', 1, 10, 3
    )

    gmm = GaussianMixture(covariance_type = 'diag', n_components = n_components, random_state = 42) 
    gmm.fit(X_pca)  

    cluster_labels = gmm.predict(X_pca)

    data_with_clusters = X_pca.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    # Show the resulting clusters
    st.write("Clustering result:")
    st.write(data_with_clusters)

    # Optionally, display a message about the clustering
    st.write(f'GMM clustered the data into {n_components} clusters.')


    cluster_counts = data_with_clusters['Cluster'].value_counts().sort_index()

    # Display the cluster counts
    st.write(f'GMM clustered the data into {n_components} clusters.')
    st.write("Count of data points in each cluster:")
    st.write(cluster_counts)

    # Optionally, show a diagram or chart for cluster counts
    show_diag = st.checkbox('Show Diagram')
    if show_diag:
        st.bar_chart(cluster_counts)

show_diag = st.checkbox('Show Diagram')








