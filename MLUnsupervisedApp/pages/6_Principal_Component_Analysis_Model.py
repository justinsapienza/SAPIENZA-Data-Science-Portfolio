import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction

# Set up the Streamlit app title
st.title("Interactive Hierarchical Clustering")

# Section 1: Upload Data or Use Sample
st.header("Upload Your Own Dataset or Use Sample")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Load the dataset, either from user upload or use the Iris dataset as a default
if uploaded_file:
    data = pd.read_csv(uploaded_file)  # Read the uploaded CSV file into a Pandas DataFrame
    st.write("File Uploaded Successfully!")
else:
    st.write("Using Sample Data: Iris Dataset")
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display dataset preview
st.subheader("Data Preview")
st.write(data.head())

# Step 1: Handling Missing Values
# Use SimpleImputer to replace missing values with the column mean
imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Step 2: Encoding Categorical Features
# Identify categorical columns
categorical_columns = data_imputed.select_dtypes(include=["object"]).columns

if len(categorical_columns) > 0:
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data_imputed[col] = label_encoders[col].fit_transform(data_imputed[col])

# Section 2: Feature Selection
st.header("Select Features for Clustering")

# Allows users to select which features they want to use for clustering
selected_columns = st.multiselect("Choose columns:", data.columns.tolist(), default=data.columns[:2].tolist())

# Extract only the selected features for clustering
data_selected = data[selected_columns]

# Step 3: Standardizing the Data
scaler = StandardScaler()  # Initialize a scaler to normalize feature values
data_scaled = scaler.fit_transform(data_selected)  # Standardize the selected features to have zero mean and unit variance

# Step 4: Hyperparameter Selection
st.header("Adjust Hierarchical Clustering Parameters")

# User selects a linkage method, which determines how clusters are merged
linkage_method = st.selectbox("Select Linkage Method", ["single", "complete", "average"])

# Step 5: Compute and Visualize Dendrogram
st.header("Dendrogram Visualization")

# Generate hierarchical clustering dendrogram based on user-selected parameters
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(data_scaled, method=linkage_method))
st.pyplot(fig)  # Display the dendrogram in the Streamlit app

st.write("""
### Understanding the Dendrogram:
- Each **horizontal line** represents a cluster merging event.
- The **height** of a merge indicates the similarity between clusters.
- **Cutting the dendrogram at a chosen level** determines the final number of clusters.
""")

# Step 6: Select Number of Clusters
st.subheader("Choose the Number of Clusters")

# User selects the number of clusters dynamically via a slider
num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

# Apply Agglomerative Hierarchical Clustering based on selected parameters
model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
clusters = model.fit_predict(data_scaled)  # Fit the model and assign cluster labels
data["Cluster"] = clusters  # Attach cluster labels to the dataset

# Display assigned clusters in the dataset
st.subheader("Cluster Assignments")
st.write(data)  # Show dataset with assigned cluster labels

# Step 7: Cluster Visualization
st.subheader("Cluster Visualization")

# Create a scatter plot to visualize clustered data points
fig, ax = plt.subplots()
scatter = ax.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)

# Add labels to the axes
ax.set_xlabel(selected_columns[0])  # Label x-axis using selected feature name
ax.set_ylabel(selected_columns[1])  # Label y-axis using selected feature name

# Create a legend to show the cluster labels
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)  # Add legend to the plot

# Display plot in Streamlit
st.pyplot(fig)

# Step 8: Cluster Visualization
st.subheader("Cluster Visualization")

# Check if more than two features were selected
if len(selected_columns) > 2:
    st.write("Since more than two features were selected, multiple plots will visualize clusters for all features.")

    # Create pairwise scatter plots for all feature combinations
    num_features = len(selected_columns)
    fig, axes = plt.subplots(num_features, num_features, figsize=(12, 12))

    for i in range(num_features):
        for j in range(num_features):
            if i != j:  # Only plot when features are different
                ax = axes[i, j]
                ax.scatter(data_selected.iloc[:, i], data_selected.iloc[:, j], c=clusters, cmap='viridis', alpha=0.6)
                ax.set_xlabel(selected_columns[i])
                ax.set_ylabel(selected_columns[j])

    st.pyplot(fig)

else:
    # Create standard scatter plot for two selected features
    fig, ax = plt.subplots()
    ax.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)  # Plot data points colored by cluster assignment
    ax.set_xlabel(selected_columns[0])  # Label x-axis using selected feature name
    ax.set_ylabel(selected_columns[1])  # Label y-axis using selected feature name
    st.pyplot(fig)

st.write("""
### Understanding the Clustering Visualization:
- Each data point is assigned to a cluster based on similarity.
- Different linkage methods produce varying clustering patterns.
- Adjusting the **number of clusters** allows you to explore different partitioning approaches.
""")

st.write("Try adjusting the number of clusters and linkage method to analyze different clustering behaviors.")