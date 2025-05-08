import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Initialize the Streamlit app with a title
st.title("K-Means Clustering Tutorial")

# Section 1: Explanation of K-Means Clustering
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

# Section 2: Real-World Applications of K-Means Clustering
st.header("Real-World Applications")
st.write("""
K-Means is widely used in various domains, including:
- **Customer Segmentation** - Grouping customers based on purchasing behavior.
- **Image Compression** - Reducing image complexity by clustering pixel values.
- **Sports Analytics** - Identifying different types of players based on performance.
- **Urban Mobility Analysis** - Classifying traffic patterns.
""")

# Section 3: Interactive Practice for K-Means Clustering
st.header("Interactive K-Means Demo")
st.write("Use the interactive elements below to explore K-Means clustering.")

## Step 1: Generate Sample Data
st.subheader("Step 1: Generate Sample Data")
# Allow user to select the number of data points interactively
n_samples = st.slider("Number of data points:", 100, 1000, 300)

# Generate synthetic data using make_blobs
X, _ = make_blobs(n_samples=n_samples, centers=10, random_state=42)

# Inform the user about the effect of sample size
st.write("Adjust the number of data points using the slider to see how it affects clustering.")

## Step 2: Determine Optimal K using the Elbow Method
st.subheader("Step 2: Elbow Method for Optimal K Selection")
wcss = []  # List to store Within-Cluster Sum of Squares (WCSS) values
K_range = range(2, 11)  # Possible values of K (number of clusters)

# Iterate over different values of K to compute WCSS
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Initialize K-Means
    kmeans.fit(X)  # Fit the model to the dataset
    wcss.append(kmeans.inertia_)  # Store the sum of squared distances for this K

# Plot the Elbow Method results
fig, ax = plt.subplots()
ax.plot(K_range, wcss, marker='o')
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
ax.set_title("Elbow Method for Optimal K")
st.pyplot(fig)

# Explanation of the elbow method
st.write("""
### Understanding the Elbow Method:
- The **Within-Cluster Sum of Squares** measures how compact clusters are. Lower values indicate better clustering.
- The "elbow" in the graph represents the point where additional clusters **no longer significantly reduce WCSS**.
- Before the elbow, reducing WCSS is **rapid**, meaning more clusters improve the separation of data points.
- After the elbow, **diminishing returns** occur, meaning additional clusters do not improve the clustering quality much.
""")

## Step 3: Select K for Clustering
st.subheader("Step 3: Select K for Clustering")
# Let the user select the number of clusters interactively
k_clusters = st.slider("Select K (number of clusters):", 2, 10, 3)
st.write("Use the slider to choose a value for **K**, the number of clusters.")

## Step 4: Apply K-Means Clustering
st.subheader("Step 4: Apply K-Means Clustering")
kmeans = KMeans(n_clusters=k_clusters, random_state=42)  # Initialize K-Means with the selected K
y_kmeans = kmeans.fit_predict(X)  # Assign each data point to a cluster
st.write("The algorithm has assigned each data point to one of the **K clusters**.")

## Step 5: Compute Silhouette Score
st.subheader("Step 5: Evaluate Clustering Quality with Silhouette Score")
sil_score = silhouette_score(X, y_kmeans)  # Compute the silhouette score
st.write(f"Silhouette Score for K={k_clusters}: **{sil_score:.4f}** (Higher is better)")
st.write("""
### Interpretation of Silhouette Score:
- Measures how well-separated the clusters are.
- Values closer to **1** indicate well-defined clusters.
- Values closer to **0** indicate overlapping clusters.
""")

## Step 6: Visualize the Clusters
st.subheader("Step 6: Visualizing the K-Means Clustering")
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', label="Data Points")  # Plot data points colored by cluster
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker="X", c="red", label="Centroids")  # Plot centroids
# Create a legend based on the scatter plot
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
# Add the label for centroids
ax.legend(["Centroids"], loc="upper right")
st.pyplot(fig)

# Explanation of the visualized clusters
st.write("""
### Understanding the Visualization:
- Each point is assigned to a cluster based on similarity.
- The **centroids (red X marks)** represent the center of each cluster.
- The clustering result changes as you modify **K**.
""")

st.write("Try adjusting the number of clusters using the slider above to see how it affects clustering.")