# app.py
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

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

    # Get the count of data points in each cluster
    cluster_counts = data_with_clusters['Cluster'].value_counts().sort_index()

    # Display the cluster counts
    st.write("Count of data points in each cluster:")
    st.write(cluster_counts)

    show_pairplot = st.checkbox('Show Pairplot')
    if show_pairplot:
        # Create a DataFrame for pairplot
        df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(5)])
        df['Cluster'] = cluster_labels

        # Plot pairplot
        plt.figure(figsize=(10, 8))
        sns.pairplot(df, hue='Cluster', palette='viridis')
        
        buf = BytesIO()
        sns_plot.savefig(buf, format="png")
        buf.seek(0)

        st.image(buf)

    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    calinski_harabasz_avg = calinski_harabasz_score(X_pca, cluster_labels)
    st.write(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.2f}")

    davies_bouldin_avg = davies_bouldin_score(X_pca, cluster_labels)
    st.write(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")








