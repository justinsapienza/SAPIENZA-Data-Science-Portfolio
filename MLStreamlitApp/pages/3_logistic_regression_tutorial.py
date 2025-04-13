import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Title of the Tutorial
st.title("Logistic Regression Tutorial")
st.write("""
This tutorial introduces Logistic Regression, a powerful classification algorithm. Learn its concepts, 
mathematics, and practical implementation with an interactive example.
""")

# Section 1: What is Logistic Regression?
st.header("What is Logistic Regression?")
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
""")

# Section 2: Interactive Demonstration
st.header("Interactive Demonstration")

# Generate a synthetic dataset
st.subheader("Step 1: Generate a Sample Dataset")
st.write("""
We will create a dataset with two features and a binary target variable to simulate a classification problem.
""")
np.random.seed(42)
X = np.random.rand(200, 2) * 10  # Two features: Random numbers between 0 and 10
y = (X[:, 0] + X[:, 1] > 10).astype(int)  # Binary target based on a simple rule

# Display the dataset
data = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
data["Target"] = y
st.dataframe(data.head())

# Section 3: Split Data and Train Model
st.subheader("Step 2: Train-Test Split and Model Training")
st.write("""
We'll split the dataset into training and testing sets, train a logistic regression model on the training data, 
and evaluate its performance on the testing data.
""")

# Train-test split
test_size = st.slider("Test Size (Percentage)", 10, 50, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Display coefficients
st.write("The model has been trained. Below are the coefficients:")
coefficients = pd.DataFrame({"Feature": ["Intercept"] + ["Feature 1", "Feature 2"],
                             "Coefficient": [model.intercept_[0]] + list(model.coef_[0])})
st.dataframe(coefficients)

# Predictions
st.subheader("Step 3: Make Predictions and Evaluate the Model")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)

st.write("Performance Metrics:")
st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"AUC Score: {auc_score:.2f}")

# ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
st.pyplot(plt)

st.write("""
This concludes the tutorial! You learned about logistic regression, trained a model, 
and evaluated its performance using various metrics and visualizations.
""")