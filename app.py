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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import Birch

# Title of the app
st.title('FB Live Seller Clustering')

url = 'https://raw.githubusercontent.com/longjoey/Clustering/main/X_pca2.csv'
url2 = 'https://raw.githubusercontent.com/longjoey/Clustering/main/normalized_df2.csv'
url3 = 'https://raw.githubusercontent.com/longjoey/Clustering/main/X_pca3.csv'
url4 = 'https://raw.githubusercontent.com/longjoey/Clustering/main/normalized_df3.csv'

def load_data():
    return pd.read_csv(url)

def load_data2():
    return pd.read_csv(url2)

def load_data3():
    return pd.read_csv(url3)

def load_data4():
    return pd.read_csv(url4)

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

def plot_gmm_clusters(X_pca, cluster_labels, n_components):
    if isinstance(X_pca, pd.DataFrame):
        X_pca = X_pca.to_numpy()
        
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, edgecolor='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'GMM Clustering Results ({n_components} Components)')
    plt.colorbar(label='Cluster')
    plt.grid()

    # Use Streamlit's st.pyplot() to display the plot
    st.pyplot(plt)
    plt.close()  # Close the plot to avoid display issues

def plot_pairplot(X_pca, cluster_labels):
    # Create a DataFrame for pairplot
    if isinstance(X_pca, pd.DataFrame):
        df = X_pca.copy()
    else:
        df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(4)])
    
    df['Cluster'] = cluster_labels

    # Plot pairplot
    plt.figure(figsize=(10, 8))
    pairplot = sns.pairplot(df, hue='Cluster', palette='viridis')
    plt.suptitle('Pairplot of PCA Components Colored by Clusters', y=1.02)
    
    # Use Streamlit's st.pyplot() to display the plot
    st.pyplot(plt)
    plt.close()  # Close the plot to avoid display issues

def plot_dbscan_clusters(X_pca, cluster_labels, n_components):
    if isinstance(X_pca, pd.DataFrame):
        X_pca = X_pca.to_numpy()
        
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='plasma')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('DBSCAN Clustering on PCA-Reduced Data')
    cbar = plt.colorbar(scatter, ax=ax, label='Cluster Label')


    # Use Streamlit's st.pyplot() to display the plot
    st.pyplot(plt)
    plt.close()  # Close the plot to avoid display issues



X_pca = load_data()
normalized_df = load_data2()

# If the user inputs their name, show a greeting
model = st.selectbox(
    'Select a clustering model:',
    ('GMM', 'Agglomerative Clustering', 'BIRCH Clustering', 'DBSCAN', 'Self-Organizing Maps')
)

if model == 'GMM' :
    n_components = st.slider(
        'Number of components:', 1, 10, 8
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

    
    plot_pairplot(X_pca, cluster_labels)
     

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

    plot_dbscan_clusters(X_pca, cluster_labels, min_samples)
    

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

    filtered_data = X_pca[cluster_labels != -1]
    filtered_labels = cluster_labels[cluster_labels != -1]

    silhouette_avg = silhouette_score(filtered_data, filtered_labels)
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    calinski_harabasz_avg = calinski_harabasz_score(filtered_data, filtered_labels)
    st.write(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.2f}")

    davies_bouldin_avg = davies_bouldin_score(filtered_data, filtered_labels)
    st.write(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")

if model == 'Self-Organizing Maps':
    # Convert DataFrame to NumPy array
    X_pca = X_pca.values  # This is critical to ensure compatibility with MiniSom
    
    # Sliders for SOM grid dimensions
    som_x = st.slider('Select SOM grid width:', 5, 20, 10)  # Grid width
    som_y = st.slider('Select SOM grid height:', 5, 20, 10)  # Grid height
    
    # Initialize and train the SOM
    som = MiniSom(x=som_x, y=som_y, input_len=X_pca.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X_pca)  # Initialize weights with input data
    som.train_random(X_pca, 1000)  # Train the SOM

  
 
    st.subheader("SOM Clustering Visualization")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for i, x in enumerate(X_pca):
        w = som.winner(x)  # Get the winning neuron
        ax1.text(w[0] + 0.5, w[1] + 0.5, str(i), color=plt.cm.tab10(i % 10), fontdict={'weight': 'bold', 'size': 9})
    
    ax1.set_xlim([0, som_x])
    ax1.set_ylim([0, som_y])
    ax1.set_title('SOM Clustering on PCA-Reduced Data')
    st.pyplot(fig1)
    
    # Visualize the SOM weight distance map (U-Matrix)
    st.subheader("SOM U-Matrix Visualization")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    cax = ax2.pcolor(som.distance_map().T, cmap='bone_r')  # Distance map as background
    fig2.colorbar(cax, label='Distance')
    ax2.set_title('SOM U-Matrix')
    st.pyplot(fig2)

    winner_coordinates = np.array([som.winner(x) for x in X_pca])

    # Convert the coordinates into a single cluster label for each data point
    som_cluster_labels = np.ravel_multi_index(winner_coordinates.T, dims=(som_x, som_y))

    # Convert the data and labels to a DataFrame for easier analysis
    df = pd.DataFrame(X_pca, columns=[f'Feature {i}' for i in range(X_pca.shape[1])])
    df['Cluster'] = som_cluster_labels

    # Calculate the number of data points in each cluster
    cluster_counts = df['Cluster'].value_counts().sort_index()

    # Display the counts
    st.write("Number of data points in each cluster:")
    st.write(cluster_counts)

    # Calculate mean and median statistics for each cluster
    mean_stats = df.groupby('Cluster').mean()
    median_stats = df.groupby('Cluster').median()

    # Display the mean and median statistics
    st.write("Mean statistics for each cluster:")
    st.write(mean_stats)
    
    st.write("Median statistics for each cluster:")
    st.write(median_stats)

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

if model == 'Agglomerative Clustering':
    X_pca = load_data3()
    normalized_df = load_data4()
    
    dataset = st.selectbox(
        'Select a dataset:',
        ('Normal', 'Dimentionality Reduction')
    )
    n_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=5)
    if dataset == 'Normal':

        # Perform Agglomerative Clustering
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        cluster_labels = hc.fit_predict(normalized_df)
                
        # Scatter plot visualization (adjust indices to match your dataset)
        st.write('Scatter plot of Clusters')
        fig, ax = plt.subplots()
        scatter = ax.scatter(normalized_df.iloc[:, 0], normalized_df.iloc[:, 1], c=cluster_labels, cmap='viridis')
        plt.title(f'Scatter plot of Clusters (n_clusters={n_clusters})')
        plt.xlabel('Feature 0')  # Replace with actual feature name
        plt.ylabel('Feature 1')  # Replace with actual feature name
        fig.colorbar(scatter, label='Cluster')
        st.pyplot(fig)

        # Calculate Silhouette Score
        silhouette_avg = silhouette_score(normalized_df, cluster_labels)
        st.write(f'Silhouette Score for {n_clusters} clusters: {silhouette_avg}')

    else:
        # Step 2: Perform Agglomerative Clustering on PCA-transformed data
        agglo = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        labels = agglo.fit_predict(X_pca)

        # Plot the clusters
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Check if X_pca is a DataFrame or a NumPy array
        if isinstance(X_pca, pd.DataFrame):
            ax.scatter(X_pca.iloc[:, 0], X_pca.iloc[:, 1], c=labels, cmap='jet', marker='h')
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='jet', marker='h')
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title(f'Agglomerative Clustering on PCA-transformed Data (n_clusters={n_clusters})')
        
        # Add a colorbar to the plot
        fig.colorbar(ax.scatter(X_pca.iloc[:, 0] if isinstance(X_pca, pd.DataFrame) else X_pca[:, 0], 
                                X_pca.iloc[:, 1] if isinstance(X_pca, pd.DataFrame) else X_pca[:, 1], 
                                c=labels, cmap='jet', marker='h'), label='Cluster')
        
        # Show the plot in Streamlit
        st.pyplot(fig)

        silhouette_agglo = silhouette_score(X_pca, labels)
        calinski_agglo = calinski_harabasz_score(X_pca, labels)

        st.write(f"**Silhouette Score**: {silhouette_agglo:.4f}")
        st.write(f"**Calinski-Harabasz Index**: {calinski_agglo:.4f}")
        

if model == 'BIRCH Clustering':
    X_pca = load_data3()
    normalized_df = load_data4()

    dataset = st.selectbox(
        'Select a dataset:',
        ('Normal', 'Dimentionality Reduction')
    )
    
    n_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=5)
    threshold = st.slider('Select threshold value for BIRCH clustering', min_value=0.1, max_value=1.0, value=0.7)
    
    if dataset == 'Normal':
    
        # Step 1: Apply BIRCH clustering
        birch_model = Birch(n_clusters=n_clusters, threshold=threshold)
        birch_labels = birch_model.fit_predict(normalized_df)
        
        # Step 2: Visualize the clusters using the first two features
        st.subheader("BIRCH Clustering Visualization")
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(normalized_df.iloc[:, 0], normalized_df.iloc[:, 1], c=birch_labels, cmap='viridis')
        ax.set_xlabel('Feature 1 (scaled)')
        ax.set_ylabel('Feature 2 (scaled)')
        ax.set_title(f'BIRCH Clustering Visualization (n_clusters={n_clusters}, threshold={threshold})')
        
        # Add a colorbar to indicate cluster assignments
        fig.colorbar(scatter, label='Cluster')
        
        # Display the plot in Streamlit
        st.pyplot(fig)

        silhouette_avg = silhouette_score(normalized_df, birch_labels)
        st.write(f'Silhouette Score for {n_clusters} clusters: {silhouette_avg}')
        
        
    else:
        birch_model = Birch(n_clusters=n_clusters, threshold=threshold)
        birch_labels = birch_model.fit_predict(X_pca)
        
        # Plot the clusters
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Check if X_pca is a DataFrame or a NumPy array
        if isinstance(X_pca, pd.DataFrame):
            ax.scatter(X_pca.iloc[:, 0], X_pca.iloc[:, 1], c=birch_labels, cmap='jet', marker='h')
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=birch_labels, cmap='jet', marker='h')
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title(f'Birch Clustering on PCA-transformed Data (n_clusters={n_clusters})')
        
        # Add a colorbar to the plot
        fig.colorbar(ax.scatter(X_pca.iloc[:, 0] if isinstance(X_pca, pd.DataFrame) else X_pca[:, 0], 
                                X_pca.iloc[:, 1] if isinstance(X_pca, pd.DataFrame) else X_pca[:, 1], 
                                c=birch_labels, cmap='jet', marker='h'), label='Cluster')
        
        # Show the plot in Streamlit
        st.pyplot(fig)


        silhouette_birch = silhouette_score(X_pca, birch_labels)
        calinski_birch = calinski_harabasz_score(X_pca, birch_labels)

        st.write(f"**Silhouette Score**: {silhouette_birch:.4f}")
        st.write(f"**Calinski-Harabasz Index**: {calinski_birch:.4f}")

 


        
            

