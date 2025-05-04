import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Set up the app title
st.title("K-Means Clustering Tutorial")

# Section 1: Theory
st.header("📌 Understanding K-Means Clustering")
st.write("""
K-Means is an **unsupervised learning algorithm** used to partition data into **K clusters** based on similarity.

### Key Steps:
1. Choose the number of clusters **K**.
2. Initialize **K centroids** randomly.
3. Assign each data point to its **nearest centroid**.
4. Update centroids by taking the mean of all points in each cluster.
5. Repeat until centroids stabilize.
""")

# Display formula for centroid update
st.latex(r"c_k = \frac{1}{N_k} \sum_{i=1}^{N_k} x_i")

# Section 2: Algorithm Breakdown
st.header("🔍 Algorithm Breakdown")
st.write("""
1️⃣ **Select K**: Define the number of clusters.  
2️⃣ **Random Initialization**: Pick K random centroids.  
3️⃣ **Assignment Step**: Each point joins its closest centroid.  
4️⃣ **Update Step**: Centroids shift to the average position.  
5️⃣ **Convergence**: Repeats until centroids no longer move.
""")

# Section 3: Real-World Applications
st.header("🌍 Real-World Applications")
st.write("""
- **Customer Segmentation** (Marketing)
- **Image Compression** (Computer Vision)
- **Grouping Sports Fans** (Cultural Analytics)
- **Urban Mobility Analysis** (Traffic Planning)
""")

# Section 4: Interactive Practice
st.header("⚡ Interactive K-Means Demo")

# Generate sample data
n_samples = st.slider("Number of data points:", 100, 1000, 300)
k_clusters = st.slider("Select K (number of clusters):", 2, 10, 3)

X, _ = make_blobs(n_samples=n_samples, centers=k_clusters, random_state=42)
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plot results
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker="X", c="red", label="Centroids")
ax.legend()
st.pyplot(fig)

st.write("🔹 Try adjusting the number of clusters using the slider above.")