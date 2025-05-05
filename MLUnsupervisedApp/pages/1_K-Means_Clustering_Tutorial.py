import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Set up the app title
st.title("K-Means Clustering Tutorial")

# Section 1: Understanding K-Means Clustering
st.header("Understanding K-Means Clustering")
st.write("""
K-Means is an **unsupervised learning algorithm** used to partition data into **K clusters** based on similarity.

### Key Steps:
1. Choose the number of clusters **K**.
2. Initialize **K centroids** randomly.
3. Assign each data point to its **nearest centroid**.
4. Update centroids by taking the mean of all points in each cluster.
5. Repeat until centroids stabilize.
""")

# Section 2: Real-World Applications
st.header("Real-World Applications")
st.write("""
- **Customer Segmentation** (Marketing)
- **Image Compression** (Computer Vision)
- **Grouping Sports Fans** (Cultural Analytics)
- **Urban Mobility Analysis** (Traffic Planning)
""")

# Section 3: Interactive Practice
st.header("Interactive K-Means Demo")
st.write("Use the interactive elements below to explore K-Means clustering.")

## Step 1: Generate Sample Data
st.subheader("Step 1: Generate Sample Data")
n_samples = st.slider("Number of data points:", 100, 1000, 300)
X, _ = make_blobs(n_samples=n_samples, centers=10, random_state=42)
st.write("Adjust the number of data points using the slider to see how it affects clustering.")

## Step 2: Determine Optimal K using the Elbow Method
st.subheader("Step 2: Elbow Method for Optimal K Selection")
wcss = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(K_range, wcss, marker='o')
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
ax.set_title("Elbow Method for Optimal K")
st.pyplot(fig)

st.write("""
🔍 **How the Elbow Method Works**:
- The elbow in the graph represents the point where adding more clusters **no longer significantly reduces WCSS**.
- Before the elbow, reducing WCSS is **rapid** as more clusters improve separation.
- After the elbow, **diminishing returns** make additional clusters **less useful**.
- The **optimal K** is where the elbow appears, ensuring a good balance between **accuracy and simplicity**.
""")

## Step 3: Select K for Clustering
st.subheader("Step 3: Select K for Clustering")
k_clusters = st.slider("Select K (number of clusters):", 2, 10, 3)
st.write("Use the slider to choose a value for **K**, the number of clusters.")

## Step 4: Apply K-Means Clustering
st.subheader("Step 4: Apply K-Means Clustering")
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X)
st.write("The algorithm has assigned each data point to one of the **K clusters**.")

## Step 5: Compute Silhouette Score
st.subheader("Step 5: Evaluate Clustering Quality with Silhouette Score")
sil_score = silhouette_score(X, y_kmeans)
st.write(f"Silhouette Score for K={k_clusters}: **{sil_score:.4f}** (Higher is better)")
st.write("A high **silhouette score** indicates well-separated and cohesive clusters.")

## Step 6: Visualize the Clusters
st.subheader("Step 6: Visualize the K-Means Clustering")
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker="X", c="red", label="Centroids")
ax.legend()
st.pyplot(fig)
st.write("Each point is assigned to a cluster, and centroids are marked in red.")

st.write("Try adjusting the number of clusters using the slider above to see how it affects clustering.")