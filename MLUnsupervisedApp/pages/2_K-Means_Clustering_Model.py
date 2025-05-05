import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

# Set up the Streamlit app title
st.title("Interactive K-Means Clustering")

# Section 1: Upload Data or Use Sample
st.header("Upload Your Own Dataset or Use Sample")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Load the dataset, either from user upload or use the Iris dataset as a default
if uploaded_file:
    data = pd.read_csv(uploaded_file)  # Read the uploaded CSV file into a Pandas DataFrame
    st.write("File Uploaded Successfully!")  # Notify the user of successful upload
else:
    st.write("Using Sample Data: Iris Dataset")  # Use the Iris dataset if no file is uploaded
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Preview the first few rows of the dataset
st.subheader("Data Preview")
st.write(data.head())  # Display initial dataset rows for user reference

# Section 2: Feature Selection
st.header("Select Features for Clustering")
selected_columns = st.multiselect("Choose columns:", data.columns.tolist(), default=data.columns[:2].tolist())

# Extract only the selected features for clustering
data_selected = data[selected_columns]

# Step 3: Standardizing the Data
scaler = StandardScaler()  # Create a standard scaler instance
data_scaled = scaler.fit_transform(data_selected)  # Scale the selected data to have zero mean and unit variance

# Step 4: Determine Optimal K using the Elbow Method
st.subheader("Elbow Method for Optimal K Selection")

wcss = []  # List to store within-cluster sum of squares (WCSS) values
K_range = range(2, 11)  # Possible values for K (number of clusters)

# Run K-Means clustering for different values of K and calculate WCSS
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Initialize K-Means with K clusters
    kmeans.fit(data_scaled)  # Fit the K-Means model to the scaled data
    wcss.append(kmeans.inertia_)  # Store the sum of squared distances for this K

# Plot the WCSS values for different K values
fig, ax = plt.subplots()
ax.plot(K_range, wcss, marker='o')  # Plot the elbow method graph
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
ax.set_title("Elbow Method for Optimal K")
st.pyplot(fig)  # Display the plot in the Streamlit app

# Explanation of the elbow method
st.write("""
**How the Elbow Method Works**:
- The elbow in the graph represents the point where adding more clusters **no longer significantly reduces WCSS**.
- Before the elbow, reducing WCSS is **rapid**, meaning more clusters improve separation of data points.
- After the elbow, **diminishing returns** occur, meaning additional clusters do not improve clustering quality significantly.
- The **optimal K** is where the elbow appears, ensuring a balance between **accuracy and simplicity**.
""")

# Step 5: Hyperparameter Selection
st.header("Adjust K-Means Parameters")

# User selects the number of clusters via a slider
k_clusters = st.slider("Select K (Number of Clusters)", 2, 10, 3)

# User chooses centroid initialization method
init_method = st.selectbox("Centroid Initialization Method", ["k-means++", "random"])

# User selects the maximum number of iterations for K-Means
max_iter = st.slider("Max Iterations", 100, 500, 300)

# Step 6: Apply K-Means Clustering
kmeans = KMeans(n_clusters=k_clusters, init=init_method, max_iter=max_iter, random_state=42)  # Initialize K-Means with user-defined parameters
clusters = kmeans.fit_predict(data_scaled)  # Apply K-Means clustering to the scaled data

# Step 7: Compute Silhouette Score
sil_score = silhouette_score(data_scaled, clusters)  # Compute silhouette score to assess clustering quality
st.subheader("Silhouette Score for Cluster Cohesion")
st.write(f"Silhouette Score for K={k_clusters}: **{sil_score:.4f}** (Higher is better)")  # Display the silhouette score

# Explanation of silhouette score
st.write("""
**Interpreting the Silhouette Score**:
- Measures how well-separated the clusters are.
- Values closer to **1** indicate well-defined clusters.
- Values closer to **0** indicate overlapping clusters.
""")

# Step 8: Visualize the Clustering Results
st.header("Clustering Results")

# Create a scatter plot of clustered data points
fig, ax = plt.subplots()
ax.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)  # Plot the clustered data points
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker="X", c="red", label="Centroids")  # Highlight cluster centroids
ax.set_xlabel(selected_columns[0])  # Label x-axis with the selected feature name
ax.set_ylabel(selected_columns[1])  # Label y-axis with the selected feature name
ax.legend()  # Add a legend to indicate centroids
st.pyplot(fig)  # Display the plot in Streamlit

# Explanation of the visualized clusters
st.write("""
**Understanding the Clustering Visualization**:
- Each point represents an observation assigned to a cluster.
- The **centroids (red X marks)** indicate the calculated center of each cluster.
- Different choices for **K** result in different clustering outcomes.
""")

st.write("Try adjusting the parameters and explore the clustering results!")