import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# App Title
st.title("Principal Component Analysis (PCA) Tutorial")

# Section 1: Understanding PCA
st.header("Understanding Principal Component Analysis")
st.write("""
Principal Component Analysis (PCA) is a **dimensionality reduction** technique used to simplify complex datasets while preserving as much relevant information as possible.

### Why use PCA?
- High-dimensional data can be **difficult to analyze and visualize**. PCA reduces the complexity while maintaining essential structures.
- By transforming data into a new set of **orthogonal axes (principal components)**, PCA **removes correlations** and identifies dominant patterns.
""")

# Section 2: Real-World Applications
st.header("Applications of PCA")
st.write("""
- **Retail Analytics** – Identifying customer spending trends and clustering behaviors  
- **Cultural Analysis** – Discovering hidden trends in urban interactions and preferences  
- **Financial Modeling** – Recognizing economic patterns in stock movements  
- **Sports Analytics** – Differentiating playing styles based on statistical metrics  
""")

# Section 3: Interactive PCA Demo

## Step 1: Generate Synthetic Data
st.subheader("Step 1: Demo Dataset - Customer Purchasing Behavior")
st.write("This dataset represents various purchasing behaviors including spending amount, store visits, discount usage, product categories purchased, and customer loyalty.")

np.random.seed(42)
data = pd.DataFrame({
    "Monthly Spending ($)": np.random.randint(50, 1000, 50),
    "Number of Store Visits": np.random.randint(1, 30, 50),
    "Discount Usage (%)": np.random.randint(0, 100, 50),
    "Product Categories Purchased": np.random.randint(1, 10, 50),
    "Customer Loyalty Score": np.random.randint(1, 100, 50)
})

st.write(data.head())

## Step 2: Feature Selection
st.subheader("Step 2: Choose Features for PCA")
st.write("Select the variables to include in PCA. PCA works best when features are numeric and continuous.")

selected_columns = st.multiselect("Choose features:", data.columns.tolist(), default=data.columns.tolist())

## Step 3: Standardizing the Data
st.subheader("Step 3: Standardizing the Data")
st.write("Since PCA relies on variance, standardizing ensures all features contribute equally.")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[selected_columns])

## Step 4: Selecting Number of Principal Components
st.subheader("Step 4: Select Number of Principal Components")
st.write("""
PCA transforms the dataset into **principal components**, each representing a combination of original features.  
Choose how many principal components to retain based on explained variance.
""")

n_components = st.slider("Select Number of Principal Components:", 1, len(selected_columns), 2)

## Step 5: Apply PCA
st.subheader("Step 5: Applying PCA")
st.write("PCA identifies the most meaningful directions (principal components) in the data, reducing redundancy.")

pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(data_scaled)

# Create DataFrame for Principal Components
pc_df = pd.DataFrame(principal_components, columns=[f"PC {i+1}" for i in range(n_components)])

## Step 6: Visualizing Explained Variance
st.subheader("Step 6: Understanding Explained Variance")
st.write("""
Each principal component explains a portion of the dataset’s variance.  
The explained variance ratio indicates how much information is preserved in each component.
""")

fig, ax = plt.subplots()
ax.bar(range(1, n_components+1), pca.explained_variance_ratio_, color='skyblue')
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("Variance Explained by Each Component")
st.pyplot(fig)

## Step 7: Scatter Plot of Principal Components
st.subheader("Step 7: Principal Components Scatter Plot")
st.write("""
Visualizing data in the principal component space helps reveal **clusters, trends, or relationships** in a reduced dimensional form.
""")

if n_components >= 2:
    fig, ax = plt.subplots()
    sns.scatterplot(x=pc_df.iloc[:, 0], y=pc_df.iloc[:, 1], alpha=0.7)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Data Projection onto First Two Principal Components")
    st.pyplot(fig)

st.write("Adjust the number of principal components and explore how PCA transforms the data.")