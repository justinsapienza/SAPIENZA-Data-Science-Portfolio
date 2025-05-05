import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# App Title
st.title("Interactive Principal Component Analysis (PCA)")

# Section 1: Upload Data or Use Sample
st.header("📂 Upload Your Dataset or Use Sample")
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
st.header("📊 Select Features for PCA")
selected_columns = st.multiselect("Choose columns:", data.columns.tolist(), default=data.columns.tolist())

# Standardize Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[selected_columns])

# Select Number of Principal Components
n_components = st.slider("Select Number of Principal Components:", 1, len(selected_columns), 2)

# Apply PCA
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(data_scaled)

# Create DataFrame for Principal Components
pc_df = pd.DataFrame(principal_components, columns=[f"PC {i+1}" for i in range(n_components)])

# Visualization
st.header("📈 PCA Explained Variance")
fig, ax = plt.subplots()
ax.bar(range(1, n_components+1), pca.explained_variance_ratio_, color='skyblue')
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("Variance Explained by Each Component")
st.pyplot(fig)

st.subheader("🌐 Principal Components Scatter Plot")
if n_components >= 2:
    fig, ax = plt.subplots()
    sns.scatterplot(x=pc_df.iloc[:, 0], y=pc_df.iloc[:, 1], alpha=0.7)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Data Projection onto First Two Principal Components")
    st.pyplot(fig)

st.write("🔹 Adjust the number of principal components and explore how PCA transforms the data!")