import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Set up the Streamlit app title
st.title("Interactive Hierarchical Clustering Tutorial")

# Section 1: Explanation of Hierarchical Clustering
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

# Section 2: Real-World Applications of Hierarchical Clustering
st.header("Applications of Hierarchical Clustering")
st.write("""
Hierarchical clustering is widely applied in various fields:
- **Social Networks** - Identifying community structures in online interactions.
- **Market Segmentation** - Categorizing customers based on purchase behavior.
- **Genetic Research** - Classifying biological organisms based on genetic similarity.
- **Urban Cultural Analysis** - Understanding community engagement patterns.
""")

# Section 3: Interactive Practice for Hierarchical Clustering
st.header("Interactive Hierarchical Clustering Demo")

## Step 1: Load or Generate Sample Data
st.subheader("Step 1: Demo Dataset - Customer Spending Habits")

# Generate synthetic dataset representing customer behavior
np.random.seed(42)
data = pd.DataFrame({
    "Customer ID": range(1, 51),
    "Annual Income (k$)": np.random.randint(15, 120, 50),
    "Spending Score (1-100)": np.random.randint(1, 100, 50)
})

# Display the first few rows of the dataset
st.write(data.head())

## Step 2: Feature Selection
st.subheader("Step 2: Select Features for Clustering")
selected_columns = st.multiselect("Choose features:", data.columns[1:], default=["Annual Income (k$)", "Spending Score (1-100)"])

# Extract only the selected features for clustering
data_selected = data[selected_columns]

## Step 3: Standardizing the Data
scaler = StandardScaler()  # Create a standard scaler instance
data_scaled = scaler.fit_transform(data_selected)  # Scale the selected data to ensure equal weighting of features

## Step 4: Selecting a Linkage Method
st.subheader("Step 3: Choose a Linkage Method")
linkage_method = st.radio("Select Linkage Method", ["single", "complete", "average"])
st.write("""
**Linkage determines how clusters are merged:**
- **Single Linkage:** Uses the shortest distance between clusters.
- **Complete Linkage:** Uses the longest distance between clusters.
- **Average Linkage:** Uses the mean distance between clusters.
""")

## Step 5: Compute and Visualize Dendrogram
st.subheader("Step 4: Dendrogram Visualization")

# Generate and display dendrogram based on selected linkage method
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(data_scaled, method=linkage_method))
st.pyplot(fig)

st.write("""
### Understanding the Dendrogram:
- Each **horizontal line** represents a clustering merge.
- The **height** of a merge indicates how far apart clusters were.
- **Cutting the dendrogram** at a chosen level determines the number of clusters.
""")

## Step 6: Select Number of Clusters and Apply Clustering
st.subheader("Step 5: Cluster Assignments")

# User selects the number of clusters via a slider
num_clusters = st.slider("Choose Number of Clusters", 2, 10, 3)

# Apply hierarchical clustering based on selected parameters
model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
clusters = model.fit_predict(data_scaled)
data["Cluster"] = clusters  # Assign cluster labels to dataset

# Display assigned clusters
st.write("Cluster assignments:")
st.write(data)

## Step 7: Cluster Visualization
st.subheader("Step 6: Cluster Visualization")

# Create a scatter plot of clustered data points
fig, ax = plt.subplots()
ax.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)  # Plot clustered data points
ax.set_xlabel(selected_columns[0])  # Label x-axis with selected feature name
ax.set_ylabel(selected_columns[1])  # Label y-axis with selected feature name
st.pyplot(fig)

st.write("""
### Understanding the Clustering Visualization:
- Each point represents a customer assigned to a cluster.
- Different choices for **number of clusters and linkage methods** result in different clustering outcomes.
""")

st.write("Try adjusting the number of clusters and linkage method to see how it affects clustering!")