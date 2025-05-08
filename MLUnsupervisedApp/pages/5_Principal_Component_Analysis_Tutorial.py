import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction

# Set up the Streamlit app title
st.title("Interactive Hierarchical Clustering")

# Introduction
st.write("""
## Understanding PCA
**Principal Component Analysis (PCA)** is a dimensionality reduction technique that:
- Finds the best direction for **maximum data variance**.
- Compresses data while **preserving essential patterns**.
- Helps in **visualizing high-dimensional datasets**.
""")

# Key Concepts in PCA
st.write("""
## Finding the Best Direction
- PCA identifies the direction in which the **data spreads the most**, meaning **maximum variance**.
- These directions, called **principal components**, form new axes that capture differences in the dataset.

## Variance Explained
- Each principal component explains a **portion** of the total variance.
- The **first principal component** captures **the highest variance**, while subsequent components contribute **progressively less**.

## Rotation and Compression
- PCA **rotates** the dataset to align with these new principal component axes.
- It **compresses** the data while preserving the most **meaningful information**, removing redundant features.
""")

# Section 1: Demo Dataset Creation
st.header("Interactive PCA Demo")

# Generate synthetic customer data for clustering
np.random.seed(42)  # Ensure reproducibility
data = pd.DataFrame({
    "Customer ID": range(1, 51),  # Unique customer IDs
    "Annual Income (k$)": np.random.randint(15, 120, 50),  # Simulated annual income
    "Spending Score (1-100)": np.random.randint(1, 100, 50)  # Simulated spending score
})

# Preview the first few rows of the demo dataset
st.subheader("Demo Data Preview")
st.write(data.head())  # Display sample dataset rows

# Section 2: Feature Selection
st.header("Select Features for Clustering")

# Enables users to choose features for clustering analysis
selected_columns = st.multiselect("Choose columns:", data.columns[1:], default=["Annual Income (k$)", "Spending Score (1-100)"])

# Extract only the selected features for clustering
data_selected = data[selected_columns]

# Step 3: Standardizing the Data
scaler = StandardScaler()  # Initialize the scaler for feature normalization
data_scaled = scaler.fit_transform(data_selected)  # Apply normalization to standardize feature values

# Step 4: Hyperparameter Selection
st.header("Adjust Hierarchical Clustering Parameters")

# User selects a linkage method, which determines how clusters are merged
linkage_method = st.selectbox("Select Linkage Method", ["single", "complete", "average"])

# Explanation of linkage methods
st.write("""
### Understanding Linkage Methods:
- **Single Linkage (Minimum Distance)**  
  - Merges clusters based on the shortest distance between individual points.  
  - Can lead to **long chains of connected points**, instead of well-defined groups.  

- **Complete Linkage (Maximum Distance)**  
  - Merges clusters using the farthest distance between points.  
  - Creates **compact and evenly distributed clusters**, avoiding elongated shapes.  

- **Average Linkage (Mean Distance)**  
  - Merges clusters using the average distance between all points.  
  - Balances between single and complete linkage, creating **moderate-sized clusters**.  
""")

# Step 5: Compute and Visualize Dendrogram (Truncated)
st.header("Dendrogram Visualization")

# Generate a truncated dendrogram for hierarchical clustering based on selected parameters
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram = sch.dendrogram(
    sch.linkage(data_scaled, method=linkage_method),  # Perform hierarchical clustering
    truncate_mode="level",  # Truncate dendrogram for better readability
    p=10,  # Limit the number of levels shown
    show_leaf_counts=True  # Display the number of points per cluster
)
st.pyplot(fig)  # Display dendrogram in Streamlit

st.write("""
### Understanding the Dendrogram:
- Each **horizontal line** represents a cluster merging event.
- The **height** of a merge indicates the similarity between clusters.
- The dendrogram is **truncated** to improve readability while keeping essential clustering information.
""")

# Step 6: Select Number of Clusters
st.subheader("Choose the Number of Clusters")

# User sets the number of clusters dynamically via slider
num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

# Apply Agglomerative Hierarchical Clustering based on selected parameters
model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
clusters = model.fit_predict(data_scaled)  # Assign cluster labels to data points
data["Cluster"] = clusters  # Attach cluster labels to the dataset

# Display assigned clusters in the dataset
st.subheader("Cluster Assignments")
st.write(data)

# Step 7: Cluster Visualization with Legend
st.subheader("Cluster Visualization")

# Create a scatter plot to visualize clustered data points
fig, ax = plt.subplots()
scatter = ax.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1], c=clusters, cmap="viridis", alpha=0.6, label="Data Points")  # Color-coded data points
centroids = ax.scatter(
    np.mean(data_selected.iloc[:, 0]), np.mean(data_selected.iloc[:, 1]), 
    s=200, marker="X", c="red", label="Centroids"  # Red Xs mark estimated centroids
)

# Add a legend to clarify clusters and centroids
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")  # Create cluster legend
ax.add_artist(legend1)  # Display cluster legend
ax.legend(["Centroids"], loc="upper right", title="Cluster Centers")  # Additional centroid legend

# Label axes with selected feature names
ax.set_xlabel(selected_columns[0])  # Label x-axis
ax.set_ylabel(selected_columns[1])  # Label y-axis

st.pyplot(fig)

st.write("""
### Understanding the Clustering Visualization:
- Each data point is assigned to a cluster based on similarity.
- Different linkage methods produce varying clustering patterns.
- The **red X markers** represent estimated **centroids**, helping to understand cluster centers.
""")

# Step 8: PCA Visualization for High-Dimensional Data (Only when more than two features are selected)
if len(selected_columns) > 2:
    st.subheader("PCA Projection for High-Dimensional Data")

    # Explanation of PCA Visualization
    st.write("""
    ### What Does the PCA Plot Show?
    - Since more than two features were selected, this plot **reduces the dimensions** using **Principal Component Analysis (PCA)**.
    - PCA simplifies the dataset by projecting data onto two principal components.
    - It helps visualize clusters in a **two-dimensional space**, making it easier to interpret high-dimensional data.
    """)

    # Apply PCA to reduce dimensionality to 2 components
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(data_scaled)  # Transform scaled data into two PCA components

    # Create a PCA scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=clusters, cmap="viridis", alpha=0.6, label="Data Points")  # Color points by cluster
    centroids = ax.scatter(
        np.mean(pca_transformed[:, 0]), np.mean(pca_transformed[:, 1]), 
        s=200, marker="X", c="red", label="Centroids"  # Red Xs mark estimated centroids
    )

    # Add a legend to clarify clusters and centroids
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")  # Create cluster legend
    ax.add_artist(legend1)  # Display cluster legend
    ax.legend(["Centroids"], loc="upper right", title="Cluster Centers")  # Additional centroid legend

    # Label axes as PCA components
    ax.set_xlabel("Principal Component 1")  # Label x-axis
    ax.set_ylabel("Principal Component 2")  # Label y-axis

    st.pyplot(fig)  # Display PCA plot in Streamlit

st.write("Try adjusting the number of clusters and linkage method to analyze different clustering behaviors.")