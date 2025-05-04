Unsupervised Machine Learning Project
=====================================
Unsupervised machine learning finds patterns and structures in data without explicit labels. Unsupervised learning explores data on its own, uncovering hidden relationships and groupings.

Project Overview
----------------
The goal of this project is to use unsupervised machine learning to explore hidden patterns in data using three type of methodologies: **k-means clustering**, **hierarchical clustering**, and **principal component analysis**. My app is an interactive learning tool that educates users on the unsupervised learning techniques and allows for hands-on practice. Here's what users can do with it:

📍**Tutorials**
- understand the theory behind each unsupervised model
- learn key equations, algorithms, and visualization techniques
- explore real-world applications

📍**Models**
- upload your own dataset or use the preloaded dataset
- adjust hyperparamters (number of clusters, etc.)
- execute each model and visualize results

Instructions
------------
Step-by-step instructions on how to run the app locally and a link to the deployed version!
1. Create a python file
2. Import the streamlit library
```python
import streamlit as st
```
3. Right click on the file and select "Copy Relative Path"
4. Open the terminal in the bottom right corner
5. Type into the terminal "streamlit run" and paste the relative path
6. Hit enter
7. The app will open as a local host in your browser

Here is a link to the deployed version(insert link here) of the app!

App Features
------------
📁 Hyperparamter Selection 

K-Means Clustering
- Select features for clustering
- Adjust number of clusters
- Choose centroid initialization method
- Select the max iterations
 
Hierarchical Clustering
- Select features for clustering
- Select linkage method
- Select distance metric
  
Principal Component Analysis
- Select features for PCA
- Select number of principal components
  
📊 Visualizations

K-Means Clustering
- clusters graph
- (Need to add elbow plots and silhouette scores)
  
Hierarchical Clustering
- Dendrogram

Principal Component Analysis
- Variance Histogram
- Scatterplot

References
----------
See additional references that informed my app creation

- [K-Means Clustering Overview](https://www.geeksforgeeks.org/k-means-clustering-introduction/)
- [K-Means Clustering Algorithm](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)
- [Hierarchical Clustering Overview](https://www.geeksforgeeks.org/hierarchical-clustering/)
- [PCA Overview](https://www.geeksforgeeks.org/principal-component-analysis-pca/)
- [PCA Step-by-Step Guide](https://www.turing.com/kb/guide-to-principal-component-analysis)
