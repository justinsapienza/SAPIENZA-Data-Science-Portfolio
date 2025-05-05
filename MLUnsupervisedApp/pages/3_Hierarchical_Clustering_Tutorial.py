import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler

# App Title
st.title("Hierarchical Clustering Tutorial")

# Section 1: Theory
st.header("📌 Understanding Hierarchical Clustering")
st.write("""
Hierarchical clustering is an **unsupervised learning method** that organizes data into nested clusters, forming a **tree-like structure (dendrogram)**.

### 🔍 Two Approaches:
- **Agglomerative (Bottom-Up)** – Each point starts as its own cluster, then merges progressively.
- **Divisive (Top-Down)** – All points start as one cluster, then split iteratively.

### 🔑 Key Components:
1️⃣ **Distance Calculation** – Measures similarity between points (e.g., Euclidean distance).  
2️⃣ **Linkage Methods** – Defines how clusters are merged:
   - **Single Linkage:** Uses the closest pair of points.
   - **Complete Linkage:** Uses the farthest pair of points.
   - **Average Linkage:** Takes the mean distance.  
3️⃣ **Dendrogram Visualization** – Helps determine the number of meaningful clusters.
""")

# Display formula for distance calculation
st.latex(r"d(i, j) = \sqrt{\sum (x_i - x_j)^2}")

# Section 2: Algorithm Breakdown
st.header("🔍 Algorithm Steps")
st.write("""
1️⃣ Compute distance between all data points.  
2️⃣ Merge the two closest clusters.  
3️⃣ Update distances & repeat until all points merge into one cluster.  
4️⃣ **Cut the dendrogram** to define clusters.
""")

# Section 3: Real-World Applications
st.header("🌍 Applications of Hierarchical Clustering")
st.write("""
- **Social Networks** – Identifying communities within online platforms.  
- **Market Segmentation** – Categorizing customers based on behavior.  
- **Genetic Research** – Classifying biological organisms.  
- **Urban Cultural Analysis** – Exploring city-based engagement trends.
""")

# Section 4: Interactive Practice
st.header("⚡ Interactive Hierarchical Clustering Demo")

# Made-Up Dataset: Simulated Customer Spending Habits
st.subheader("📝 Demo Dataset: Customer Spending Habits")
np.random.seed(42)
data = pd.DataFrame({
    "Customer ID": range(1, 51),
    "Annual Income (k$)": np.random.randint(15, 120, 50),
    "Spending Score (1-100)": np.random.randint(1, 100, 50)
})
st.write(data.head())

# Feature Selection
selected_columns = ["Annual Income (k$)", "Spending Score (1-100)"]
data_selected = data[selected_columns]

# Standardize Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Select Linkage Method
linkage_method = st.selectbox("Select Linkage Method", ["single", "complete", "average"])

# Compute Hierarchical Clustering & Dendrogram
st.subheader("📈 Dendrogram Visualization")
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(data_scaled, method=linkage_method))
st.pyplot(fig)

st.write("🔹 Adjust linkage method & explore clustering structures!")