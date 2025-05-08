import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Initialize the Streamlit app with a title
st.title("Interactive Hierarchical Clustering Tutorial")

# Section 1: Explanation of Hierarchical Clustering
st.header("Understanding Hierarchical Clustering")
st.write("""
Hierarchical clustering is an **unsupervised learning method** that organizes data into nested clusters, forming a **tree-like structure (dendrogram)**.

### Two Approaches:
- **Agglomerative (Bottom-Up)** – Each point starts as its own cluster, then merges progressively.
- **Divisive (Top-Down)** – All points start as one cluster, then split iteratively.
""")

# Section 2: Real-World Applications
st.header("Applications of Hierarchical Clustering")
st.write("""
Hierarchical clustering is widely applied in various fields:
- **Social Networks** - Identifying community structures in online interactions.
- **Market Segmentation** - Categorizing customers based on purchase behavior.
""")

# Step 1: Generate or Load Data
st.subheader("Step 1: Demo Dataset - Customer Spending Habits")

# Generate synthetic dataset representing customer behavior
np.random.seed(42)
data = pd.DataFrame({
    "Customer ID": range(1, 51),  # Unique customer IDs
    "Annual Income (k$)": np.random.randint(15, 120, 50),  # Simulated annual income
    "Spending Score (1-100)": np.random.randint(1, 100, 50)  # Simulated spending score
})

# Display the first few rows for user reference
st.write("Sample Data Preview:")
st.write(data.head())

# Step 2: Feature Selection
st.subheader("Step 2: Select Features for Clustering")
selected_columns = st.multiselect("Choose features:", data.columns[1:], default=["Annual Income (k$)", "Spending Score (1-100)"])

# Extract selected features for clustering
data_selected = data[selected_columns]

# Step 3: Standardize Data
scaler = StandardScaler()  # Normalize data for better clustering
data_scaled = scaler.fit_transform(data_selected)  # Ensures equal weighting across features

# Step 4: Choose Linkage Method with Explanation
st.subheader("Step 3: Choose a Linkage Method")
linkage_method = st.radio("Select Linkage Method", ["single", "complete", "average"])

# Explanation of linkage methods in a tutorial format
st.write("""
### Understanding Linkage Methods:
- **Single Linkage (Minimum Distance)**  
  - Merges clusters based on the **closest** pair of points.  
  - Can result in **long, chain-like clusters**, which may not always be well-separated.  

- **Complete Linkage (Maximum Distance)**  
  - Merges clusters based on the **most distant** pair of points.  
  - Produces **compact and evenly distributed clusters**, avoiding elongated shapes.  

- **Average Linkage (Mean Distance)**  
  - Uses the **average** distance between all points in two clusters.  
  - Balances between single and complete linkage, creating **moderate-sized clusters**.  
""")

# Step 5: Compute and Visualize Dendrogram (Truncated)
st.subheader("Step 4: Dendrogram Visualization")

# Generate hierarchical structure using selected linkage method
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram = sch.dendrogram(
    sch.linkage(data_scaled, method=linkage_method),
    truncate_mode='level',  # Simplifies tree visualization
    p=10,  # Limits levels shown for clarity
    show_leaf_counts=True  # Displays cluster sizes
)
st.pyplot(fig)

st.write("""
### Understanding the Truncated Dendrogram:
- Each horizontal line represents a **merge** between clusters.
- The height of a merge indicates **distance between clusters**.
- Cutting the dendrogram at an appropriate level helps define **optimal clusters**.
""")

# Step 6: Apply Hierarchical Clustering
st.subheader("Step 5: Cluster Assignments")

# User selects the number of clusters
num_clusters = st.slider("Choose Number of Clusters", 2, 10, 3)

# Apply clustering model using selected linkage method
model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
clusters = model.fit_predict(data_scaled)

# Store cluster labels in the dataset for visualization
data["Cluster"] = clusters

# Show cluster assignments
st.write("Cluster Assignments Preview:")
st.write(data)

# Step 7: Visualize Clusters with Legend
st.subheader("Step 6: Cluster Visualization")

# Generate scatter plot of clustered data points
fig, ax = plt.subplots()
scatter = ax.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6, label="Data Points")

# Estimate centroids by calculating mean feature values
centroids = ax.scatter(np.mean(data_selected.iloc[:, 0]), np.mean(data_selected.iloc[:, 1]), s=100, marker="X", c="red", label="Centroids")

# Add a legend explaining clusters and centroids
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.legend(["Centroids"], loc="upper right", title="Cluster Centers")

# Label axes with selected feature names
ax.set_xlabel(selected_columns[0])
ax.set_ylabel(selected_columns[1])

# Display plot in Streamlit
st.pyplot(fig)

st.write("""
### Understanding the Cluster Visualization:
- Each point represents a customer, assigned to a **cluster** based on similarity.
- The **centroids (red X markers)** represent estimated cluster centers.
""")