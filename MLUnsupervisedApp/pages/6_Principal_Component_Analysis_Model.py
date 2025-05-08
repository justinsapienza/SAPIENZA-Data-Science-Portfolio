import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris

# Set up the Streamlit app
st.title("Interactive PCA Model")

# Step 1: Upload Dataset or Use Sample Data
st.header("Upload Your Dataset or Use Sample")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load user-uploaded dataset
    data = pd.read_csv(uploaded_file)
else:
    # Load sample Iris dataset
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display dataset preview
st.subheader("Data Preview")
st.write(data.head())

# Step 2: Preprocess Data (Handle Missing Values & Encode Categorical Variables)
imputer = SimpleImputer(strategy="mean")  # Handle missing data
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

categorical_columns = data_imputed.select_dtypes(include=["object"]).columns
if len(categorical_columns) > 0:
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        data_imputed[col] = label_encoders[col].fit_transform(data_imputed[col])

# Step 3: Feature Selection for PCA
st.header("Select Features for PCA")
selected_columns = st.multiselect("Choose features:", data_imputed.columns.tolist(), default=data_imputed.columns[:3].tolist())

# Extract selected features
data_selected = data_imputed[selected_columns]

# Step 4: Standardizing the Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Step 5: Apply PCA (Auto-detect number of components)
pca = PCA()
data_pca = pca.fit_transform(data_scaled)

# Step 6: Visualizing Explained Variance
st.subheader("Explained Variance by Principal Components")
explained_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots()
ax.bar(range(1, len(explained_variance) + 1), explained_variance, color="blue")
ax.set_xlabel("Principal Components")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("Variance Distribution Across Principal Components")
ax.legend(["Explained Variance"])
st.pyplot(fig)

# Step 7: PCA Projection Visualization
st.subheader("PCA Projection")

if data_pca.shape[1] == 1:
    # Histogram visualization when only one principal component is retained
    fig, ax = plt.subplots()
    ax.hist(data_pca[:, 0], bins=30, color="blue", alpha=0.7)
    ax.set_xlabel(f"Principal Component (from {selected_columns[0]})")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Data on Principal Component (Derived from {selected_columns[0]})")
    ax.legend([f"PC1 - {selected_columns[0]}"])
    st.pyplot(fig)

elif data_pca.shape[1] == 2:
    # Scatter plot for 2 principal components with dynamic labels
    fig, ax = plt.subplots()
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.6, cmap="viridis", label="Data Points")
    ax.set_xlabel(f"Principal Component 1 (Derived from {selected_columns[0]})")
    ax.set_ylabel(f"Principal Component 2 (Derived from {selected_columns[1]})")
    ax.set_title(f"PCA Projection: {selected_columns[0]} vs {selected_columns[1]}")
    ax.legend([f"PC1 - {selected_columns[0]}", f"PC2 - {selected_columns[1]}"])
    st.pyplot(fig)

elif data_pca.shape[1] > 2:
    # Pairplot visualization for multiple components with feature mapping
    st.subheader("Pairplot of Principal Components")
    pca_df = pd.DataFrame(data_pca, columns=[f"PC{i+1} ({selected_columns[i]})" for i in range(data_pca.shape[1])])

    # Create scatterplot matrix for dimensional analysis
    pairplot_fig = sns.pairplot(pca_df)
    st.pyplot(pairplot_fig.fig)

st.write("Try selecting different features for PCA and observe the explained variance.")