import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Title of the Tutorial
st.title("Linear Regression Tutorial")
st.write("""
Welcome to the Linear Regression Tutorial! This page will help you understand how linear regression works 
by walking through its concepts, mathematical formulation, and an interactive example.
""")

# Section 1: Explanation of Linear Regression
st.header("What is Linear Regression?")
st.write("""
Linear regression is a statistical method used to model the relationship between a dependent variable (target) 
and one or more independent variables (features) using a straight line. The general formula for simple linear regression is:

**y = β₀ + β₁x + ε**

Where:
- **y**: Dependent variable (target value)
- **x**: Independent variable (feature)
- **β₀**: Intercept (value of y when x = 0)
- **β₁**: Slope (how much y changes for a unit change in x)
- **ε**: Error term (difference between actual and predicted values)
""")

# Section 2: Demonstration
st.header("Step-by-Step Demonstration")
st.write("""
Let's explore linear regression interactively using a sample dataset. We'll generate a simple dataset with 
a linear relationship and train a regression model to understand how it predicts values.
""")

# Generate a sample dataset
st.subheader("Step 1: Generate a Sample Dataset")
st.write("Below is a dataset with a linear relationship:")
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature: Random numbers between 0 and 10
y = 2.5 * X + np.random.randn(100, 1) * 2  # Target: Linear relationship with noise

data = pd.DataFrame({"Feature (X)": X.flatten(), "Target (y)": y.flatten()})
st.dataframe(data.head())

# Scatter plot of the dataset
st.subheader("Visualizing the Dataset")
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', alpha=0.7)
ax.set(title="Scatter Plot of the Dataset", xlabel="Feature (X)", ylabel="Target (y)")
st.pyplot(fig)

# Train-Test Split
st.subheader("Step 2: Split the Dataset")
st.write("""
To train and evaluate the model, we'll split the dataset into training and testing sets. 
Typically, 80% of the data is used for training and 20% for testing.
""")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Training Set Size: {len(X_train)} samples")
st.write(f"Testing Set Size: {len(X_test)} samples")

# Train Linear Regression Model
st.subheader("Step 3: Train the Linear Regression Model")
model = LinearRegression()
model.fit(X_train, y_train)

# Display model coefficients
st.write("The model has been trained! Below are the coefficients:")
st.write(f"Intercept (β₀): {model.intercept_[0]:.2f}")
st.write(f"Slope (β₁): {model.coef_[0][0]:.2f}")

# Predictions and Evaluation
st.subheader("Step 4: Make Predictions and Evaluate")
y_pred = model.predict(X_test)

# Plot predictions vs actual values
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='red', alpha=0.7)
ax.set(title="Predicted vs Actual Values", xlabel="Actual Values", ylabel="Predicted Values")
st.pyplot(fig)

# Calculate model performance metrics
st.write("Model Performance Metrics:")
mse = np.mean((y_test - y_pred) ** 2)
r2 = model.score(X_test, y_test)
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")

st.write("""
Congratulations! You've completed the tutorial. By understanding how linear regression models a relationship, 
you can apply it to real-world problems, such as predicting house prices, stock trends, and more.
""")