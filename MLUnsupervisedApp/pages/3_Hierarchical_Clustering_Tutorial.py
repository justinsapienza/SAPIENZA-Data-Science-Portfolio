import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# App Title
st.title("Interactive Hierarchical Clustering Tutorial")

# Section 1: Understanding Hierarchical Clustering
st.header("Understanding Hierarchical Clustering")
st.write("""
Hierarchical clustering is an **unsupervised learning method** that organizes data into nested clusters, forming a **tree-like structure (dendrogram)**.

### Two Approaches:
- **Agglomerative (Bottom-Up)** – Each point starts as its own cluster, then merges progressively.
- **Divisive (Top-Down)** – All points start as one cluster, then split iteratively.

### Key Components:
1. **Distance Calculation** – Measures similarity between points (e.g., Euclidean distance).  
2. **Linkage Methods** – Defines how clusters are merged:
   - **Single Linkage:** Uses the closest pair of points.
   - **Complete Linkage:** Uses the farthest pair of points.
   - **Average Linkage:** Takes the mean distance.  
3. **Dendrogram Visualization** – Helps determine the number of meaningful clusters.
""")

# What is a dendrogram?
st.subheader("What is a Dendrogram?")
st.write("""
A **dendrogram** is a **tree-like diagram** that represents how clusters are formed at each step. It shows:
- **Which points merge into clusters first** (bottom-up approach).
- **At what distance they merge** (height of branches).
- **Where the best cluster separations might be** (cutting the dendrogram).
""")

# Section 2: Real-World Applications
st.header("Applications of Hierarchical Clustering")
st.write("""
- **Social Networks** – Identifying communities within online platforms.  
- **Market Segmentation** – Categorizing customers based on behavior.  
- **Genetic Research** – Classifying biological organisms.  
- **Urban Cultural Analysis** – Exploring city-based engagement trends.
""")

# Section 3: Interactive Practice
st.header("Interactive Hierarchical Clustering Demo")

## Step 1: Load or Generate Sample Data
st.subheader("Step 1: Demo Dataset - Customer Spending Habits")
np.random.seed(42)
data = pd.DataFrame({
    "Customer ID": range(1, 51),
    "Annual Income (k$)": np.random.randint(15, 120, 50),
    "Spending Score (1-100)": np.random.randint(1, 100, 50)
})
st.write(data.head())

## Step 2: Feature Selection
st.subheader("Step 2: Select Features for Clustering")
selected_columns = st.multiselect("Choose features:", data.columns[1:], default=["Annual Income (k$)", "Spending Score (1-100)"])
data_selected = data[selected_columns]

## Step 3: Standardizing the Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

## Step 4: Selecting a Linkage Method
st.subheader("Step 3: Choose a Linkage Method")
linkage_method = st.radio("Select Linkage Method", ["single", "complete", "average"])
st.write("Linkage affects how clusters merge.")

## Step 5: Compute and Visualize Dendrogram
st.subheader("Step 4: Dendrogram Visualization")
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(data_scaled, method=linkage_method))
st.pyplot(fig)

st.write("Use the dendrogram to determine a reasonable number of clusters.")

## Step 6: Select Number of Clusters and Apply Clustering
st.subheader("Step 5: Cluster Assignments")
num_clusters = st.slider("Choose Number of Clusters", 2, 10, 3)

model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
clusters = model.fit_predict(data_scaled)
data["Cluster"] = clusters

st.write("Cluster assignments:")
st.write(data)

## Step 7: Cluster Visualization
st.subheader("Step 6: Cluster Visualization")
fig, ax = plt.subplots()
ax.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)
ax.set_xlabel(selected_columns[0])
ax.set_ylabel(selected_columns[1])
st.pyplot(fig)

st.write("Try adjusting the number of clusters and linkage method to see how it affects clustering!")