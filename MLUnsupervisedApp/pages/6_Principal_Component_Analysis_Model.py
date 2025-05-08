import streamlit as st  # Import Streamlit for creating interactive web apps
import pandas as pd  # Import Pandas for handling dataset operations
import numpy as np  # Import NumPy for numerical computations
import matplotlib.pyplot as plt  # Import Matplotlib for data visualization
import seaborn as sns  # Import Seaborn for advanced visualizations
from sklearn.decomposition import PCA  # Import PCA from scikit-learn for dimensionality reduction
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Import StandardScaler for normalization, LabelEncoder for categorical encoding
from sklearn.impute import SimpleImputer  # Import SimpleImputer to handle missing values
from sklearn.datasets import load_iris  # Import sample Iris dataset

# Set up the Streamlit app title
st.title("Interactive PCA Model")

# Step 1: Upload Dataset or Use Sample Data
st.header("Upload Your Dataset or Use Sample")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])  # Allow user to upload their own dataset

# Load either the uploaded dataset or the sample Iris dataset
if uploaded_file:
    data = pd.read_csv(uploaded_file)  # Load user-uploaded CSV file into a DataFrame
else:
    iris = load_iris()  # Load the Iris dataset from scikit-learn
    data = pd.DataFrame(iris.data, columns=iris.feature_names)  # Convert the Iris dataset into a DataFrame

# Display the first few rows of the dataset for user reference
st.subheader("Data Preview")
st.write(data.head())

# Step 2: Preprocess Data (Handle Missing Values & Encode Categorical Variables)
imputer = SimpleImputer(strategy="mean")  # Initialize an imputer to replace missing values with the column mean
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)  # Apply imputation and convert the result back to DataFrame

# Identify categorical columns for encoding
categorical_columns = data_imputed.select_dtypes(include=["object"]).columns

if len(categorical_columns) > 0:
    # Apply Label Encoding to categorical features if found
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        data_imputed[col] = label_encoders[col].fit_transform(data_imputed[col])  # Convert categorical values into numerical values

# Step 3: Feature Selection for PCA
st.header("Select Features for PCA")
selected_columns = st.multiselect("Choose features:", data_imputed.columns.tolist(), default=data_imputed.columns[:3].tolist())  # Allow user to select multiple features for PCA

# Extract selected features for analysis
data_selected = data_imputed[selected_columns]

# Step 4: Standardizing the Data
scaler = StandardScaler()  # Initialize StandardScaler for feature normalization
data_scaled = scaler.fit_transform(data_selected)  # Apply normalization to scale features

# Step 5: Apply PCA (Auto-detect number of components)
pca = PCA()  # Initialize PCA model without a predefined number of components
data_pca = pca.fit_transform(data_scaled)  # Apply PCA transformation to scaled data

# Step 6: Visualizing Explained Variance
st.subheader("Explained Variance by Principal Components")
explained_variance = pca.explained_variance_ratio_  # Extract explained variance ratio from PCA components

# Create a bar plot to show variance captured by each principal component
fig, ax = plt.subplots()
ax.bar(range(1, len(explained_variance) + 1), explained_variance, color="blue")  # Plot variance distribution across principal components
ax.set_xlabel("Principal Components")  # Label x-axis
ax.set_ylabel("Explained Variance Ratio")  # Label y-axis
ax.set_title("Variance Distribution Across Principal Components")  # Set title of the graph
ax.legend(["Explained Variance"])  # Add a legend to describe the plot
st.pyplot(fig)  # Display the plot in Streamlit

# Step 7: PCA Projection Visualization
st.subheader("PCA Projection")

# Determine the number of principal components retained and adjust visualization accordingly
if data_pca.shape[1] == 1:
    # Histogram visualization when only one principal component is retained
    fig, ax = plt.subplots()
    ax.hist(data_pca[:, 0], bins=30, color="blue", alpha=0.7)  # Create a histogram for single principal component
    ax.set_xlabel(f"Principal Component (from {selected_columns[0]})")  # Label x-axis dynamically based on selected feature
    ax.set_ylabel("Frequency")  # Label y-axis
    ax.set_title(f"Distribution of Data on Principal Component (Derived from {selected_columns[0]})")  # Set title dynamically
    ax.legend([f"PC1 - {selected_columns[0]}"])  # Add legend for component source
    st.pyplot(fig)  # Display the histogram

elif data_pca.shape[1] == 2:
    # Scatter plot visualization for two principal components with meaningful labels
    fig, ax = plt.subplots()
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.6, cmap="viridis", label="Data Points")  # Scatter plot for two principal components
    ax.set_xlabel(f"Principal Component 1 (Derived from {selected_columns[0]})")  # Label x-axis dynamically
    ax.set_ylabel(f"Principal Component 2 (Derived from {selected_columns[1]})")  # Label y-axis dynamically
    ax.set_title(f"PCA Projection: {selected_columns[0]} vs {selected_columns[1]}")  # Set title dynamically
    ax.legend([f"PC1 - {selected_columns[0]}", f"PC2 - {selected_columns[1]}"])  # Add meaningful legend
    st.pyplot(fig)  # Display the scatter plot

elif data_pca.shape[1] > 2:
    # Pairplot visualization for multiple principal components mapped to selected features
    st.subheader("Pairplot of Principal Components")
    pca_df = pd.DataFrame(data_pca, columns=[f"PC{i+1} ({selected_columns[i]})" for i in range(data_pca.shape[1])])  # Convert PCA-transformed data into a DataFrame

    # Create scatterplot matrix for high-dimensional analysis
    pairplot_fig = sns.pairplot(pca_df)  # Generate a pairplot for PCA components
    st.pyplot(pairplot_fig.fig)  # Display the pairplot in Streamlit

st.write("Try selecting different features for PCA and observe the explained variance.")  # Encourage user interaction