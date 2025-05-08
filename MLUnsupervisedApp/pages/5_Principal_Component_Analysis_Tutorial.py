import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Initialize the Streamlit app with a title
st.title("Principal Component Analysis (PCA) Tutorial")

# Section 1: Explanation of PCA
st.header("Understanding Principal Component Analysis")
st.write("""
**Principal Component Analysis (PCA)** is a statistical method used to reduce dimensionality while retaining key patterns. 
It simplifies datasets while preserving their essential structure.

### Key Steps:
1. **Standardize the Data** - Normalize features to ensure equal importance.
2. **Compute Covariance Matrix** - Identify relationships between variables.
3. **Find Eigenvalues and Eigenvectors** - Determine principal components.
4. **Select Principal Components** - Keep the most informative ones.
5. **Transform Data** - Project data onto the selected principal components.
""")

# Section 2: Real-World Applications of PCA
st.header("Real-World Applications")
st.write("""
PCA is widely used in various fields, including:
- **Image Compression** - Reducing storage while preserving key details.
- **Finance** - Simplifying stock market trends using fewer factors.
- **Genomics** - Analyzing gene expression patterns efficiently.
- **Speech Recognition** - Reducing noise for clear speech processing.
""")

# Section 3: Interactive Practice for PCA
st.header("Interactive PCA Demo")
st.write("Use the interactive elements below to explore PCA.")

## Step 1: Generate Sample Data
st.subheader("Step 1: Generate Sample Data")
st.write("""
To apply PCA, we first need a dataset. Here, we generate synthetic data with multiple features 
to demonstrate how PCA reduces dimensionality.
""")

# Allow user to select the number of data points interactively
n_samples = st.slider("Number of data points:", 100, 1000, 300)

# Generate synthetic data using make_blobs
X, _ = make_blobs(n_samples=n_samples, centers=5, n_features=5, random_state=42)

st.write("Sample data generated successfully!")

## Step 2: Select Number of Principal Components
st.subheader("Step 2: Select Principal Components")
st.write("""
PCA reduces the number of dimensions while retaining **most of the data’s variance**. 
Select the number of **principal components** to keep.
""")

num_components = st.slider("Select Number of Principal Components:", 1, X.shape[1], 2)
st.write(f"You have selected {num_components} principal components.")

## Step 3: Apply PCA
st.subheader("Step 3: Applying PCA")
st.write("""
Before applying PCA, it is essential to **standardize the dataset** to ensure that all features 
have equal importance. Then, we perform PCA to extract the key components.
""")

scaler = StandardScaler()  # Standardize the data
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=num_components)  # Apply PCA
X_pca = pca.fit_transform(X_scaled)

st.write("PCA transformation completed!")

# Step 4: Visualize Explained Variance
st.subheader("Step 4: Explained Variance by Principal Components")
st.write("""
Principal components are ranked based on the **amount of variance** they explain. 
The first component captures the highest variance, while others capture progressively less. 
This bar chart illustrates how much variance is explained by each principal component.
""")

explained_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots()
ax.bar(range(1, len(explained_variance) + 1), explained_variance, color='blue')
ax.set_xlabel("Principal Components")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("Variance Distribution Across Principal Components")
st.pyplot(fig)

## Step 5: Visualize PCA Projection
st.subheader("Step 5: Visualizing PCA Projection")
st.write("""
This scatter plot represents the dataset after transformation using PCA. Each point is now 
represented by its principal components instead of the original features.
""")

if num_components == 2:
    # Standard 2D scatter plot for two principal components
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, cmap="viridis")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Data Projected onto Principal Components")
    st.pyplot(fig)

elif num_components > 2:
    # Pairplot visualization if more than two principal components are selected
    st.write("""
    Since more than two principal components are selected, the pairplot below shows 
    relationships between different components.
    """)
    
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(num_components)])    
    pairplot_fig = sns.pairplot(pca_df)
    st.pyplot(pairplot_fig.fig)

# Explanation of PCA visualization
st.write("""
### Understanding the Visualization:
- Each point represents a transformed data sample.
- **Principal Component 1** captures the highest variance.
- **Principal Component 2** captures the second-highest variance.
- If more than two components are selected, pairwise relationships can be observed.
- Reducing dimensions while retaining variance helps in **data exploration and visualization**.
""")

st.write("Try adjusting the number of principal components using the slider above to see how PCA affects the dataset!")