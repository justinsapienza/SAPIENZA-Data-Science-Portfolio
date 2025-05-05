import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

# Set up the Streamlit app title
st.title("Interactive Hierarchical Clustering")

# Section 1: Upload Data or Use Sample
st.header("Upload Your Dataset or Use Sample")

# Allows user to upload their own dataset for hierarchical clustering
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Load dataset either from user upload or use the Iris dataset as a default
if uploaded_file:
    data = pd.read_csv(uploaded_file)  # Read the uploaded CSV file into a Pandas DataFrame
    st.write("File Uploaded Successfully!")  # Notify the user of successful upload
else:
    st.write("Using Sample Data: Iris Dataset")  # Default dataset if no file is uploaded
    iris = load_iris()  # Load the built-in Iris dataset from sklearn
    data = pd.DataFrame(iris.data, columns=iris.feature_names)  # Convert dataset into Pandas DataFrame

# Preview the first few rows of the dataset
st.subheader("Data Preview")
st.write(data.head())  # Display a preview of the dataset to understand its structure

# Section 2: Feature Selection
st.header("Select Features for Clustering")

# Allows users to select which features they want to use for clustering
selected_columns = st.multiselect("Choose columns:", data.columns.tolist(), default=data.columns[:2].tolist())

# Extract only the selected features for clustering
data_selected = data[selected_columns]

# Step 3: Standardizing the Data
scaler = StandardScaler()  # Initialize a scaler to normalize feature values
data_scaled = scaler.fit_transform(data_selected)  # Standardize the selected features to have zero mean and unit variance

# Step 4: Hyperparameter Selection
st.header("Adjust Hierarchical Clustering Parameters")

# User selects a linkage method, which determines how clusters are merged
linkage_method = st.selectbox("Select Linkage Method", ["single", "complete", "average"])

# User selects a distance metric to measure similarity between points
distance_metric = st.selectbox("Select Distance Metric", ["euclidean", "manhattan", "cosine"])

# Step 5: Compute and Visualize Dendrogram
st.header("Dendrogram Visualization")

# Generate hierarchical clustering dendrogram based on user-selected parameters
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(data_scaled, method=linkage_method, metric=distance_metric))
st.pyplot(fig)  # Display the dendrogram in the Streamlit app

st.write("""
### Understanding the Dendrogram:
- Each **horizontal line** represents a cluster merging event.
- The **height** of a merge indicates the similarity between clusters.
- **Cutting the dendrogram at a chosen level** determines the final number of clusters.
""")

# Step 6: Select Number of Clusters
st.subheader("Choose the Number of Clusters")

# User selects the number of clusters dynamically via a slider
num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

# Apply Agglomerative Hierarchical Clustering based on selected parameters
model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
clusters = model.fit_predict(data_scaled)  # Fit the model and assign cluster labels
data["Cluster"] = clusters  # Attach cluster labels to the dataset

# Display assigned clusters in the dataset
st.subheader("Cluster Assignments")
st.write(data)  # Show dataset with assigned cluster labels

# Step 7: Cluster Visualization
st.subheader("Cluster Visualization")

# Create a scatter plot to visualize clustered data points
fig, ax = plt.subplots()
ax.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)  # Plot data points colored by cluster assignment
ax.set_xlabel(selected_columns[0])  # Label x-axis using selected feature name
ax.set_ylabel(selected_columns[1])  # Label y-axis using selected feature name
st.pyplot(fig)

st.write("""
### Understanding the Clustering Visualization:
- Each data point is assigned to a cluster based on similarity.
- Different linkage methods produce varying clustering patterns.
- Adjusting the **number of clusters** allows you to explore different partitioning approaches.
""")

st.write("Try adjusting the number of clusters and linkage method to analyze different clustering behaviors.")