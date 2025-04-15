# Import necessary libraries
import streamlit as st  # Streamlit for interactive web applications
import pandas as pd  # Pandas for data manipulation
import numpy as np  # NumPy for numerical computations
from sklearn.model_selection import train_test_split  # Split data into training/testing sets
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve  # Metrics for model evaluation
import matplotlib.pyplot as plt  # Matplotlib for data visualization

# Title of the Tutorial
st.title("Logistic Regression Tutorial")  # Display the app title
st.write("""
This tutorial introduces Logistic Regression, a powerful classification algorithm. Learn its concepts, 
mathematics, and practical implementation with an interactive example.
""")  # Introductory text explaining the tutorial's purpose

# Section 1: What is Logistic Regression?
st.header("What is Logistic Regression?")  # Add a header to explain the algorithm
st.write("""
Logistic regression is a statistical method used for classification tasks. Unlike linear regression, 
it predicts probabilities that an observation belongs to a particular class and uses a threshold (e.g., 0.5) 
to assign class labels.

Mathematically, it models the relationship between the independent variables (features) and the probability of 
the dependent variable (outcome) using the sigmoid function:

**P(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ))**

Where:
- **P(y=1|x)** is the probability of the positive class.
- **β₀, β₁, ..., βₙ** are the model coefficients (intercept and slopes).
- **x₁, ..., xₙ** are the independent variables (features).
""")  # Explain the concept and mathematical formula of logistic regression

# Section 2: Interactive Demonstration
st.header("Interactive Demonstration")  # Add a header for the demonstration section

# Generate a synthetic dataset
st.subheader("Step 1: Generate a Sample Dataset")  # Subheader for dataset generation step
st.write("""
We will create a dataset with two features and a binary target variable to simulate a classification problem. 
This allows us to demonstrate logistic regression in a controlled environment.
""")  # Explain the purpose of dataset generation

np.random.seed(42)  # Set seed for reproducibility
X = np.random.rand(200, 2) * 10  # Generate random feature values between 0 and 10
y = (X[:, 0] + X[:, 1] > 10).astype(int)  # Create binary target using a threshold rule

# Display the dataset
st.write("""
The dataset below contains two features (`Feature 1` and `Feature 2`) and a binary target (`Target`). 
The target is 1 if the sum of the features exceeds 10, and 0 otherwise.
""")  # Description of the dataset
data = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])  # Create a dataframe for feature data
data["Target"] = y  # Add the target column to the dataframe
st.dataframe(data.head())  # Show the first few rows of the dataset

# Step 2: Train-Test Split and Model Training
st.subheader("Step 2: Train-Test Split and Model Training")  # Subheader for splitting data and training the model
st.write("""
We'll split the dataset into training and testing sets. The training set teaches the model the relationship 
between the features and the target, while the testing set evaluates the model's ability to predict unseen data.
""")  # Explain the importance of splitting the dataset

test_size = st.slider("Test Size (Percentage)", 10, 50, 20) / 100  # Slider to adjust test set size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)  # Split data into training/testing sets

st.write(f"Training Set Size: {len(X_train)} samples")  # Display the size of the training set
st.write(f"Testing Set Size: {len(X_test)} samples")  # Display the size of the testing set

# Train the logistic regression model
model = LogisticRegression()  # Initialize the logistic regression model
model.fit(X_train, y_train)  # Train the model using the training data

# Display the model's coefficients
st.write("""
The model has been trained! Below are the coefficients, which indicate the relationship between each feature 
and the probability of the target being 1:
""")  # Explain the significance of the coefficients
coefficients = pd.DataFrame({
    "Feature": ["Intercept"] + ["Feature 1", "Feature 2"],  # List of features and intercept
    "Coefficient": [model.intercept_[0]] + list(model.coef_[0])  # Corresponding coefficients from the model
})
st.dataframe(coefficients)  # Display the coefficients in a table format

# Step 3: Make Predictions and Evaluate the Model
st.subheader("Step 3: Make Predictions and Evaluate the Model")  # Subheader for predictions and evaluation
st.write("""
Using the testing set, the model makes predictions for the target variable. We'll then evaluate its performance 
using several metrics to understand how well it performs.
""")  # Explain the evaluation step

y_pred = model.predict(X_test)  # Predict class labels (0 or 1) for the testing set
y_prob = model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class (1)

# Performance Metrics
st.write("Performance Metrics:")  # Subheader for performance metrics
st.write("""
- **Accuracy**: The proportion of correct predictions (both 0 and 1). Higher accuracy indicates overall better performance.
- **Precision**: The proportion of true positives (correctly predicted 1s) out of all predicted positives. High precision means fewer false positives.
- **Recall**: The proportion of true positives out of all actual positives. High recall indicates the model identifies most actual positives.
- **AUC Score (Area Under the ROC Curve)**: Measures the model's ability to distinguish between classes (0 and 1). Higher AUC indicates better performance.
""")  # Explain the significance of each metric

accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
precision = precision_score(y_test, y_pred)  # Calculate precision
recall = recall_score(y_test, y_pred)  # Calculate recall
auc_score = roc_auc_score(y_test, y_prob)  # Calculate AUC score

st.write(f"Accuracy: {accuracy:.2f}")  # Display accuracy
st.write(f"Precision: {precision:.2f}")  # Display precision
st.write(f"Recall: {recall:.2f}")  # Display recall
st.write(f"AUC Score: {auc_score:.2f}")  # Display AUC score

# Step 4: ROC Curve
st.subheader("ROC Curve")  # Subheader for ROC curve visualization
st.write("""
The Receiver Operating Characteristic (ROC) curve illustrates the trade-off between the true positive rate 
(recall) and false positive rate for different classification thresholds. The Area Under the Curve (AUC) 
provides a summary measure of the model's ability to distinguish between classes.
""")  # Explain the significance of ROC curve

fpr, tpr, _ = roc_curve(y_test, y_prob)  # Calculate false positive rate and true positive rate for ROC curve
plt.figure(figsize=(6, 4))  # Set the figure size for the plot
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color="blue")  # Plot the ROC curve
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Plot a diagonal line for reference
plt.xlabel("False Positive Rate")  # Label the x-axis
plt.ylabel("True Positive Rate")  # Label the y-axis
plt.title("Receiver Operating Characteristic (ROC) Curve")  # Add a title to the plot
plt.legend(loc="lower right")  # Place the legend in the bottom-right corner
st.pyplot(plt)  # Display the ROC curve plot in the app

# Conclusion
st.write("""
This concludes the tutorial! You learned how to create a dataset, train a logistic regression model, 
and evaluate its performance using various metrics and visualizations. Great job!
""")  # Provide concluding remarks for the tutorial