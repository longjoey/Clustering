# app.py
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from minisom import MiniSom
import numpy as np

# Title of the app
st.title('FB Live Seller Clustering')

url = 'https://raw.githubusercontent.com/longjoey/Clustering/main/X_pca.csv'

def load_data():
    return pd.read_csv(url)

def calculate_topographic_error(som, data):
    error_count = 0
    for x in data:
        # Find the best matching unit (BMU)
        bmu = som.winner(x)
        
        # Find the second-best matching unit (2nd BMU)
        dists = np.linalg.norm(som._weights - x, axis=-1)
        sorted_indices = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
        second_bmu = (sorted_indices[0][1], sorted_indices[1][1])  # 2nd BMU
        
        # Check if BMU and 2nd BMU are adjacent on the SOM grid
        if np.abs(bmu[0] - second_bmu[0]) > 1 or np.abs(bmu[1] - second_bmu[1]) > 1:
            error_count += 1
    
    # Calculate the topographic error
    topographic_error = error_count / len(data)
    return topographic_error

def calculate_mqe(som, data):
    total_error = 0
    for x in data:
        # Find the BMU for the data point
        bmu = som.winner(x)
        # Calculate the quantization error (distance from the data point to the BMU)
        quantization_error = np.linalg.norm(x - som._weights[bmu])
        total_error += quantization_error
    
    # Calculate the mean quantization error
    mean_quantization_error = total_error / len(data)
    return mean_quantization_error

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
        df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        df['Cluster'] = cluster_labels

        # Plot pairplot
        st.write("Pairplot of PCA Components Colored by Clusters")

        fig = sns.pairplot(df, hue='Cluster', palette='viridis').fig

        # Save the figure to a BytesIO object
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        # Display the image in Streamlit
        st.image(buf, use_column_width=True)

    # Calculate mean and median statistics for each cluster
    mean_stats = data_with_clusters.groupby('Cluster').mean()
    median_stats = data_with_clusters.groupby('Cluster').median()

    st.write("Mean statistics for each cluster:")
    st.write(mean_stats)

    st.write("Median statistics for each cluster:")
    st.write(median_stats)

    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    calinski_harabasz_avg = calinski_harabasz_score(X_pca, cluster_labels)
    st.write(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.2f}")

    davies_bouldin_avg = davies_bouldin_score(X_pca, cluster_labels)
    st.write(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")

if model == 'DBSCAN':
    eps = st.slider('Select eps:', 0.01, 5.0, 0.2, step=0.01)
    min_samples = st.slider('Select min_samples:', 1, 50, 5)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_pca)

 
    data_with_clusters = X_pca.copy()
    data_with_clusters['Cluster'] = cluster_labels
    

    st.write("Clustering result:")
    st.write(data_with_clusters)
    
    # Get the count of data points in each cluster
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    
    # Display the cluster counts
    st.write("Count of data points in each cluster:")
    st.write(cluster_counts)

    # Calculate mean and median statistics for each cluster
    mean_stats = data_with_clusters.groupby('Cluster').mean()
    median_stats = data_with_clusters.groupby('Cluster').median()

    st.write("Mean statistics for each cluster:")
    st.write(mean_stats)

    st.write("Median statistics for each cluster:")
    st.write(median_stats)

    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    calinski_harabasz_avg = calinski_harabasz_score(X_pca, cluster_labels)
    st.write(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.2f}")

    davies_bouldin_avg = davies_bouldin_score(X_pca, cluster_labels)
    st.write(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")

if model == 'Self-Organizing Maps':
    # Convert DataFrame to NumPy array
    X_pca = X_pca.values  # This is critical to ensure compatibility with MiniSom
    
    # Sliders for SOM grid dimensions
    som_x = st.slider('Select SOM grid width:', 5, 20, 10)  # Grid width
    som_y = st.slider('Select SOM grid height:', 5, 20, 10)  # Grid height
    
    # Initialize and train the SOM
    st.write(f'Training SOM with grid size {som_x}x{som_y}')
    som = MiniSom(x=som_x, y=som_y, input_len=X_pca.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X_pca)  # Initialize weights with input data
    som.train_random(X_pca, 1000)  # Train the SOM

        # Visualize the SOM
    st.write("SOM Weight Map")
    
    # Create a figure to plot the SOM
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Visualize the SOM weight map with a scatter plot
    weights = som.get_weights()
    
    # Use the weight vectors to color the SOM grid
    for i in range(som_x):
        for j in range(som_y):
            w = weights[i, j]
            # Plot weight nodes with color based on the sum of the weights
            ax.plot(i + 0.5, j + 0.5, 'o', markersize=20, markeredgecolor='black', 
                    markerfacecolor=plt.cm.viridis(np.sum(w)))
    
    ax.set_xlim([0, som_x])
    ax.set_ylim([0, som_y])
    ax.set_xticks(np.arange(0, som_x + 1))
    ax.set_yticks(np.arange(0, som_y + 1))
    ax.grid(True)
    ax.set_title('SOM Weight Map')
    
    # Save the figure to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    
    # Display the image in Streamlit
    st.image(buf, use_column_width=True)
    
    winner_coordinates = np.array([som.winner(x) for x in X_pca])

    # Convert the coordinates into a single cluster label for each data point
    som_cluster_labels = np.ravel_multi_index(winner_coordinates.T, dims=(som_x, som_y))

    # Step 2: Calculate the silhouette score using the cluster labels
    silhouette_avg = silhouette_score(X_pca, som_cluster_labels)

    # Display the silhouette score in Streamlit
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    dbi = davies_bouldin_score(X_pca, som_cluster_labels)

    # Display the Davies-Bouldin Index in Streamlit
    st.write(f"Davies-Bouldin Index: {dbi:.2f}")

    topo_error = calculate_topographic_error(som, X_pca)
    st.write(f"Topographic Error: {topo_error:.2f}")

    mqe = calculate_mqe(som, X_pca)
    st.write(f"Mean Quantization Error: {mqe:.4f}")
        





