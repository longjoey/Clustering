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

url = 'https://raw.githubusercontent.com/longjoey/Clustering/main/X_pca.csv'
url2 = 'https://raw.githubusercontent.com/longjoey/Clustering/main/normalized_df.csv'

def load_data():
    return pd.read_csv(url)

def load_data2():
    return pd.read_csv(url2)

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

def plot_pairplot(X_pca, cluster_labels):
    # Create a DataFrame for pairplot
    if isinstance(X_pca, pd.DataFrame):
        df = X_pca.copy()
    else:
        df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    
    df['Cluster'] = cluster_labels

    # Plot pairplot
    plt.figure(figsize=(10, 8))
    pairplot = sns.pairplot(df, hue='Cluster', palette='viridis')
    plt.suptitle('Pairplot of PCA Components Colored by Clusters', y=1.02)
    
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

    show_pairplot = st.checkbox('Show Pairplot')
    if show_pairplot:
        plot_gmm_clusters(X_pca, cluster_labels, n_components)
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
    som = MiniSom(x=som_x, y=som_y, input_len=X_pca.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X_pca)  # Initialize weights with input data
    som.train_random(X_pca, 1000)  # Train the SOM

    show_som_map = st.checkbox('Show SOM Weight Map')

    if show_som_map:
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

if model == 'Agglomerative Clustering' or model == 'BIRCH Clustering':
    # PCA Components selection slider
    n_pca_components = st.slider('Select number of PCA components:', min_value=2, max_value=10, value=5, key='pca_components_slider')
    
    # Number of clusters selection slider
    n_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=5, key='clusters_slider')

    # Step 1: Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_pca_components)
    pca_transformed = pca.fit_transform(normalized_df)

if model == 'Agglomerative Clustering':
    
    dataset = st.selectbox(
        'Select a dataset:',
        ('Normal', 'Dimentionality Reduction')
    )
    
    if dataset == 'Normal':
        n_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=5)

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
        plt.colorbar(scatter)
        st.pyplot(fig)

        normalized_df['Cluster'] = cluster_labels

        # Group the original DataFrame by cluster labels
        cluster_stats = normalized_df.groupby('Cluster')
        
        # Calculate count for each cluster
        cluster_count = cluster_stats.size()
        st.write('Cluster Counts:')
        st.write(cluster_count)
        
        # Calculate mean statistics for each cluster
        cluster_mean = cluster_stats.mean()
        st.write('Mean Statistics for Each Cluster:')
        st.dataframe(cluster_mean)
        
        # Calculate median statistics for each cluster
        cluster_median = cluster_stats.median()
        st.write('Median Statistics for Each Cluster:')
        st.dataframe(cluster_median)
    
        # Calculate and display Silhouette Score
        silhouette_avg = silhouette_score(normalized_df, cluster_labels)
        st.write(f'Silhouette Score for {n_clusters} clusters: {silhouette_avg:.2f}')
        
        # Calculate Davies-Bouldin Index
        db_index = davies_bouldin_score(normalized_df, cluster_labels)
        st.write(f'Davies-Bouldin Index: {db_index:.2f}')
        
        # Calculate Calinski-Harabasz Index
        ch_index = calinski_harabasz_score(normalized_df, cluster_labels)
        st.write(f'Calinski-Harabasz Index: {ch_index:.2f}')
    else:
        # Step 2: Perform Agglomerative Clustering on PCA-transformed data
        agglo = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        cluster_labels = agglo.fit_predict(pca_transformed)
        
        # Add cluster labels to the original DataFrame
        normalized_df['Cluster'] = cluster_labels
        
        # Step 3: Plot the clusters
        st.write('Clusters after Scaling and PCA')
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=cluster_labels, cmap='viridis', marker='o')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'Clusters (n_clusters={n_clusters}) after PCA')
        plt.colorbar(scatter, label='Cluster')
        st.pyplot(fig)
        
        # Step 5: Cluster Count, Mean, and Median Statistics

        # Group the original DataFrame by cluster labels
        cluster_stats = normalized_df.groupby('Cluster')
        
        # Calculate count for each cluster
        cluster_count = cluster_stats.size()
        st.write('Cluster Counts:')
        st.write(cluster_count)
        
        # Calculate mean statistics for each cluster
        cluster_mean = cluster_stats.mean()
        st.write('Mean Statistics for Each Cluster:')
        st.dataframe(cluster_mean)
        
        # Calculate median statistics for each cluster
        cluster_median = cluster_stats.median()
        st.write('Median Statistics for Each Cluster:')
        st.dataframe(cluster_median)

        # Step 4: Calculate Clustering Scores
        
        # Calculate Davies-Bouldin Index
        db_index = davies_bouldin_score(pca_transformed, cluster_labels)
        st.write(f'Davies-Bouldin Index: {db_index:.2f}')
        
        # Calculate Calinski-Harabasz Index
        ch_index = calinski_harabasz_score(pca_transformed, cluster_labels)
        st.write(f'Calinski-Harabasz Index: {ch_index:.2f}')
        
        # Optional: Calculate Silhouette Score
        silhouette_avg = silhouette_score(pca_transformed, cluster_labels)
        st.write(f'Silhouette Score: {silhouette_avg:.2f}')

if model == 'BIRCH Clustering':

    dataset = st.selectbox(
        'Select a dataset:',
        ('Normal', 'Dimentionality Reduction')
    )

    if dataset == 'Normal':
        n_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=5)

        # Step 1: Apply BIRCH Clustering
        birch_model = Birch(n_clusters=n_clusters, threshold=0.8)
        birch_labels = birch_model.fit_predict(normalized_df)
        
        # Step 2: Add the BIRCH cluster labels to the DataFrame
        normalized_df['BIRCH_Cluster_Labels'] = birch_labels
        
        # Step 3: Visualize the clusters using the first two features for simplicity
        st.write('BIRCH Clustering Visualization')
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(normalized_df.iloc[:, 0], normalized_df.iloc[:, 1], c=birch_labels, cmap='viridis')
        plt.xlabel('Feature 1 (scaled)')
        plt.ylabel('Feature 2 (scaled)')
        plt.title('BIRCH Clustering Visualization')
        plt.colorbar(scatter, label='Cluster')
        st.pyplot(fig)
        
        # Display the DataFrame with the cluster labels
        st.write('DataFrame with BIRCH Cluster Labels:')
        st.dataframe(normalized_df.head())
    else:
        n_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=5)
        birch_model = Birch(n_clusters=n_clusters)
        
        # Fit the model and predict cluster labels
        birch_labels = birch_model.fit_predict(pca_transformed)
        
        # Create a DataFrame for PCA-transformed data and add the cluster labels
        clustered_df = pd.DataFrame(pca_transformed, columns=[f'PC{i+1}' for i in range(pca_transformed.shape[1])])
        clustered_df['BIRCH_Cluster'] = birch_labels
        
        # Visualize the BIRCH clusters in the PCA-reduced space
        st.write('BIRCH Clustering Visualization')
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(clustered_df['PC1'], clustered_df['PC2'], c=birch_labels, cmap='viridis', marker='o')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('BIRCH Clustering Results in PCA-Reduced Space')
        
        # Add a colorbar to the scatter plot
        cbar = plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
        
        # Display the DataFrame with the first few rows
        st.write('DataFrame with BIRCH Cluster Labels:')
        st.dataframe(clustered_df.head())

        # Calculate and display the clustering metrics
        silhouette_birch = silhouette_score(pca_transformed, birch_labels)
        calinski_birch = calinski_harabasz_score(pca_transformed, birch_labels)
        
        st.write(f"Silhouette Score: {silhouette_birch:.4f}")
        st.write(f"Calinski-Harabasz Index: {calinski_birch:.4f}")

        cluster_stats = clustered_df.groupby('BIRCH_Cluster').agg(
            count=('BIRCH_Cluster', 'size'),
            mean_pc1=('PC1', 'mean'),
            median_pc1=('PC1', 'median'),
            mean_pc2=('PC2', 'mean'),
            median_pc2=('PC2', 'median')
        ).reset_index()
        
        # Separate tables for count, mean, and median statistics
        count_table = cluster_stats[['BIRCH_Cluster', 'count']]
        mean_table = cluster_stats[['BIRCH_Cluster', 'mean_pc1', 'mean_pc2']]
        median_table = cluster_stats[['BIRCH_Cluster', 'median_pc1', 'median_pc2']]
        
        # Display the tables in Streamlit
        st.write('Cluster Counts:')
        st.dataframe(count_table)
        
        st.write('Cluster Mean Statistics:')
        st.dataframe(mean_table)
        
        st.write('Cluster Median Statistics:')
        st.dataframe(median_table)
 


        
            

