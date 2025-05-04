import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# App Title
st.title("Interactive K-Means Clustering")

# Section 1: Upload Data or Use Sample
st.header("📂 Upload Your Own Dataset or Use Sample")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("✅ File Uploaded Successfully!")
else:
    st.write("Using Sample Data: Iris Dataset")
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Preview Data
st.subheader("🔍 Data Preview")
st.write(data.head())

# Feature Selection
st.header("📊 Select Features for Clustering")
selected_columns = st.multiselect("Choose columns:", data.columns.tolist(), default=data.columns[:2].tolist())
data_selected = data[selected_columns]

# Standardize Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Hyperparameter Selection
st.header("⚙️ Adjust K-Means Parameters")
k_clusters = st.slider("Select K (Number of Clusters)", 2, 10, 3)
init_method = st.selectbox("Centroid Initialization Method", ["k-means++", "random"])
max_iter = st.slider("Max Iterations", 100, 500, 300)

# Apply K-Means
kmeans = KMeans(n_clusters=k_clusters, init=init_method, max_iter=max_iter, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Visualization
st.header("📈 Clustering Results")
fig, ax = plt.subplots()
ax.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker="X", c="red", label="Centroids")
ax.set_xlabel(selected_columns[0])
ax.set_ylabel(selected_columns[1])
ax.legend()
st.pyplot(fig)

st.write("🔹 Try adjusting the parameters and explore the clustering results!")