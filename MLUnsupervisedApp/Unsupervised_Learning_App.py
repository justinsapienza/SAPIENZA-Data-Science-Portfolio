# Import the Streamlit Libray
import streamlit as st

# Create Page Layout
st.set_page_config(page_title = "Home", layout = "centered")
st.title("Unsupervised Machine Learning Tutorial")

# Use Markdowns For App Overview
st.markdown(
    """
    Unsupervised machine learning is a type of machine learning where an algorithm is trained from data without labeled outputs. Unlike supervised learning, where the model learns from labeled outputs, unsupervised learning finds hidden patterns, structure, or relationships within the data. Unsupervised learning is valuable in exploratory data analysis, pattern discovery, and making sense of large, unstructured datasets. Since there is no explicit target variable, the insights it uncovers often reveal underlying structures that might not be obvious at first glance.

    ## Key Concepts of Unsupervised Learning

    1. Clustering: groups similar data points together based on shared characteristics.

    2. Dimensionality Reduction: simplifies complex datasets by reducing the number of features while preserving essential information.

    # Unsupervised Machine Learning Models

    ## K-Means Clustering
    K-means clustering is an unsupervised machine learning algorithim used to partition data into *k* distinct clusters based on similarity. It operates by minimizing the variance within each cluster and is commonly applied in pattern recognition, data segmentation, and anomaly detection.

    ## Hierarchical Clustering
    Hierarchical clustering is an unsupervised machine learning method used to group data points into a hierarchy of nested clusters. Unlike k-means clustering, it does not require a predefined number of clusters (*k*). Instead, it builds a tree-like structure called a **dendrogram**, allowing for flexible exploration of cluster relationships.

    ## Principal Component Analysis
    PCA is a dimensionality reduction technique used in machine learning and statistics to simplify complex datasets while preserving as much relevant information as possible. It works by transforming correlated variables into a set of uncorrelated principal components, ranking them by the amount of variance they capture.

    # App Structure

    **1. Home Page - Introduction (You are here!)**
    - Brief overview of unsupervised learning and its significance.
    - Explanation of the three models covered in the tutorials.
    - Interactive navigation to select a specific tutorial or model to practice (use the arrow in the top left corner)

    **2. Tutotials for Each Model**  
    Each model tutorial includes:
    - theory behind the model
    - real-world applications
    - a demo model

    **3. Interactive Practice Models**  
    Each model has an interactive interface, allowing users to:
    - Upload their own dataset
    - adjust model parameters
    - execute models
    """
)