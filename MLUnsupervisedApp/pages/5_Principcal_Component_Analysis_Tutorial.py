import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# App Title
st.title("Principal Component Analysis (PCA) Tutorial")

# Section 1: Theory
st.header("📌 Understanding PCA")
st.write("""
Principal Component Analysis (PCA) is a **dimensionality reduction** technique used to simplify complex datasets while preserving as much relevant information as possible.

### 🔑 Key Benefits:
✅ Reduces dimensionality while preserving essential data  
✅ Identifies dominant patterns in high-dimensional datasets  
✅ Improves visualization by mapping data onto fewer dimensions  

### 🔍 Key Steps:
1️⃣ Standardize the Data  
2️⃣ Compute Covariance Matrix  
3️⃣ Extract Eigenvectors & Eigenvalues  
4️⃣ Select Principal Components  
5️⃣ Transform Data into Principal Component Space
""")

# Display key formula for eigenvalues
st.latex(r"\lambda v = A v")

# Section 2: Algorithm Breakdown
st.header("🔍 Algorithm Steps")
st.write("""
1️⃣ Compute covariance matrix.  
2️⃣ Calculate eigenvectors and eigenvalues.  
3️⃣ Rank components by variance explained.  
4️⃣ Project data onto the principal components.  
""")

# Section 3: Real-World Applications
st.header("🌍 Applications of PCA")
st.write("""
- **Retail Analytics** – Identifying customer spending trends  
- **Cultural Analysis** – Exploring behavioral patterns in urban spaces  
- **Financial Modeling** – Understanding market movements  
- **Basketball Data Analysis** – Distinguishing playing styles  
""")

# Section 4: Interactive PCA Demo

# Generate Synthetic Customer Data
np.random.seed(42)
data = pd.DataFrame({
    "Monthly Spending ($)": np.random.randint(50, 1000, 50),
    "Number of Store Visits": np.random.randint(1, 30, 50),
    "Discount Usage (%)": np.random.randint(0, 100, 50),
    "Product Categories Purchased": np.random.randint(1, 10, 50),
    "Customer Loyalty Score": np.random.randint(1, 100, 50)
})

st.subheader("📝 Made-Up Dataset: Customer Purchasing Behavior")
st.write(data.head())

# Feature Selection
selected_columns = st.multiselect("Choose features for PCA:", data.columns.tolist(), default=data.columns.tolist())

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
st.subheader("📈 PCA Explained Variance")
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