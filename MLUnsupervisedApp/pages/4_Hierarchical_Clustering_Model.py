import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

# App Title
st.title("Interactive Hierarchical Clustering")

# Section 1: Upload Data or Use Sample
st.header("Upload Your Dataset or Use Sample")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("File Uploaded Successfully!")
else:
    st.write("Using Sample Data: Iris Dataset")
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Preview Data
st.subheader("Data Preview")
st.write(data.head())

# Feature Selection
st.header("Select Features for Clustering")
selected_columns = st.multiselect("Choose columns:", data.columns.tolist(), default=data.columns[:2].tolist())
data_selected = data[selected_columns]

# Standardize Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Hyperparameter Selection
st.header("⚙️ Adjust Hierarchical Clustering Parameters")
linkage_method = st.selectbox("Select Linkage Method", ["single", "complete", "average"])
distance_metric = st.selectbox("Select Distance Metric", ["euclidean", "manhattan", "cosine"])

# Compute Hierarchical Clustering & Dendrogram
st.header("Dendrogram Visualization")
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(data_scaled, method=linkage_method, metric=distance_metric))
st.pyplot(fig)

st.write("""
**Understanding the Dendrogram**:
- Each **horizontal line** represents a cluster merging event.
- The **height** of a merge indicates cluster similarity.
- **Cutting the dendrogram at a specific level** defines cluster boundaries.
""")

# User Interaction: Select Number of Clusters
st.subheader("Choose the Number of Clusters")
num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

# Apply Hierarchical Clustering
model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
clusters = model.fit_predict(data_scaled)
data["Cluster"] = clusters

# Display Assigned Clusters
st.subheader("Cluster Assignments")
st.write(data)

# Visualize Clustered Data
st.subheader("Cluster Visualization")
fig, ax = plt.subplots()
ax.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)
ax.set_xlabel(selected_columns[0])
ax.set_ylabel(selected_columns[1])
st.pyplot(fig)

st.write("⚡ Try adjusting the **number of clusters** and linkage method to explore clustering behavior!")