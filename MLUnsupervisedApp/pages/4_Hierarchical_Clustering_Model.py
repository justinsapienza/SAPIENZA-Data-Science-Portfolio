import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering

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

# Enables users to choose the features they want to include in clustering
selected_columns = st.multiselect("Choose columns:", data.columns.tolist(), default=data.columns[:2].tolist())

# Extract only the selected features for clustering
data_selected = data[selected_columns]

# Step 3: Standardizing the Data
scaler = StandardScaler()  # Normalize feature values to have zero mean and unit variance
data_scaled = scaler.fit_transform(data_selected)  # Ensures equal weighting of features

# Step 4: Hyperparameter Selection
st.header("Adjust Hierarchical Clustering Parameters")

# Allows user to select the linkage method, which determines how clusters merge
linkage_method = st.selectbox("Select Linkage Method", ["single", "complete", "average"])

# Explanation of linkage methods
st.write("""
### Understanding Linkage Methods:
- **Single Linkage:** Merges clusters based on the shortest distance between points.  
  - Can lead to long chains of connected points rather than compact groups.  

- **Complete Linkage:** Merges clusters based on the farthest distance between points.  
  - Creates well-separated, compact clusters that avoid chaining.  

- **Average Linkage:** Merges clusters based on the mean distance between all points.  
  - Balances between single and complete linkage for moderately compact clusters.  
""")

# Step 5: Compute and Visualize Dendrogram (Truncated)
st.header("Dendrogram Visualization")

# Generates and displays truncated dendrogram for hierarchical clustering
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram = sch.dendrogram(
    sch.linkage(data_scaled, method=linkage_method),  # Corrected comma placement
    truncate_mode="level",  # Truncates dendrogram for better readability
    p=10,  # Limits number of clusters displayed
    show_leaf_counts=True  # Displays number of points in each cluster
)
st.pyplot(fig)

st.write("""
### Understanding the Dendrogram:
- Each **horizontal line** represents a clustering merge event.
- The **height** of a merge indicates how far apart clusters were before merging.
- The dendrogram is **truncated** to improve readability while keeping essential clustering information.
""")

# Step 6: Select Number of Clusters
st.subheader("Choose the Number of Clusters")

# Allows user to dynamically set the number of clusters
num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

# Apply hierarchical clustering based on selected parameters
model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
clusters = model.fit_predict(data_scaled)  # Fit the model and assign cluster labels
data["Cluster"] = clusters  # Attach cluster labels to the dataset

# Display assigned cluster groups
st.subheader("Cluster Assignments")
st.write(data)

# Step 7: Cluster Visualization with Legend
st.subheader("Cluster Visualization")

# Creates scatter plot of clustered data points
fig, ax = plt.subplots()
scatter = ax.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1], c=clusters, cmap="viridis", alpha=0.6, label="Data Points")  # Plot data points colored by cluster

# Estimate centroids by calculating mean feature values
centroids = ax.scatter(
    np.mean(data_selected.iloc[:, 0]), np.mean(data_selected.iloc[:, 1]),
    s=200, marker="X", c="red", label="Centroids"
)

# Create a legend explaining clusters and centroids
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.legend(["Centroids"], loc="upper right", title="Cluster Centers")

# Label axes with selected feature names
ax.set_xlabel(selected_columns[0])  # Label x-axis
ax.set_ylabel(selected_columns[1])  # Label y-axis

st.pyplot(fig)

st.write("""
### Understanding the Cluster Visualization:
- Each point represents a data observation assigned to a cluster.
- The **red X marks** indicate estimated **centroids**, showing the central point of each cluster.
- Adjusting the number of clusters and linkage method reveals different clustering patterns.
""")

# Step 8: Additional Visualization (Only for more than two features)
if len(selected_columns) > 2:
    st.subheader("Advanced Visualization for Multi-Dimensional Data")

    # Explanation of the visualization
    st.write("""
    ### Understanding the Multi-Dimensional Visualization:
    - Since more than two features have been selected, we use a **pair plot** to show relationships between features.
    - Each subplot represents the scatter plot of two features, colored by cluster assignment.
    - This visualization helps in understanding how features interact across different clusters.
    """)

    # Create a DataFrame to hold scaled data and cluster assignments
    data_plot = pd.DataFrame(data_scaled, columns=selected_columns)
    data_plot["Cluster"] = clusters  # Add cluster labels for coloring

    # Generate a pair plot using Seaborn
    pairplot_fig = sns.pairplot(data_plot, hue="Cluster", palette="viridis")

    # Display the pair plot
    st.pyplot(pairplot_fig)

st.write("Try adjusting the number of clusters and linkage method to see how clustering behavior changes!")