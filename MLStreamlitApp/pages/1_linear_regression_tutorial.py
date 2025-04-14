# Import necessary libraries
import streamlit as st  # Streamlit for interactive web application
import numpy as np  # NumPy for numerical computations
import pandas as pd  # Pandas for data manipulation and analysis
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.model_selection import train_test_split  # Split data into training and testing sets

# Title of the Tutorial
st.title("Linear Regression Tutorial")  # Set the title for the web application
st.write("""
Welcome to the Linear Regression Tutorial! This page will help you understand how linear regression works 
by walking through its concepts, mathematical formulation, and an interactive example.
""")  # Introductory text explaining the purpose of the tutorial

# Section 1: Explanation of Linear Regression
st.header("What is Linear Regression?")  # Add a section header for explanation
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
""")  # Explanation of linear regression with its formula and components

# Section 2: Demonstration
st.header("Step-by-Step Demonstration")  # Add a header for demonstration section
st.write("""
Let's explore linear regression interactively using a sample dataset. We'll generate a simple dataset with 
a linear relationship and train a regression model to understand how it predicts values.
""")  # Introduction to the demonstration

# Step 1: Generate a Sample Dataset
st.subheader("Step 1: Generate a Sample Dataset")  # Subheader for dataset generation step
st.write("""
In this step, we create a synthetic dataset with a known linear relationship. This allows us to demonstrate 
linear regression in a controlled environment.
""")  # Explain the purpose of generating a sample dataset
np.random.seed(42)  # Set random seed for reproducibility
X = np.random.rand(100, 1) * 10  # Generate random numbers between 0 and 10 for feature
y = 2.5 * X + np.random.randn(100, 1) * 2  # Generate target variable with a linear relationship and added noise

data = pd.DataFrame({"Feature (X)": X.flatten(), "Target (y)": y.flatten()})  # Create a dataframe for the dataset
st.dataframe(data.head())  # Display the first few rows of the dataset

# Scatter plot of the dataset
st.subheader("Visualizing the Dataset")  # Subheader for dataset visualization
st.write("""
Below is a scatter plot of the dataset, showing the relationship between the feature (X) and the target (y).
""")  # Explain the scatter plot's purpose
fig, ax = plt.subplots()  # Create a figure and axes for the plot
ax.scatter(X, y, color='blue', alpha=0.7)  # Scatter plot of feature vs target
ax.set(title="Scatter Plot of the Dataset", xlabel="Feature (X)", ylabel="Target (y)")  # Set plot title and axis labels
st.pyplot(fig)  # Display the plot in the Streamlit app

# Step 2: Split the Dataset
st.subheader("Step 2: Split the Dataset")  # Subheader for splitting the dataset
st.write("""
To train and evaluate the linear regression model, we divide the dataset into training and testing sets. 
Typically, the training set comprises 80% of the data, which is used to teach the model, and the testing set 
comprises 20%, which is used to evaluate its predictions.
""")  # Explain the importance of splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the dataset
st.write(f"Training Set Size: {len(X_train)} samples")  # Display training set size
st.write(f"Testing Set Size: {len(X_test)} samples")  # Display testing set size

# Step 3: Train the Linear Regression Model
st.subheader("Step 3: Train the Linear Regression Model")  # Subheader for training the model
st.write("""
In this step, we train the linear regression model using the training set. The model learns the relationship 
between the feature (X) and the target (y), expressed as a mathematical equation: y = β₀ + β₁X, where β₀ 
is the intercept and β₁ is the slope.
""")  # Explain the training process
model = LinearRegression()  # Initialize the Linear Regression model
model.fit(X_train, y_train)  # Train the model using the training data

# Display model coefficients
st.write("The model has been trained! Below are the coefficients:")  # Notify the user that training is complete
st.write(f"Intercept (β₀): {model.intercept_[0]:.2f}")  # Display the model's intercept
st.write(f"Slope (β₁): {model.coef_[0][0]:.2f}")  # Display the model's slope

# Step 4: Make Predictions and Evaluate
st.subheader("Step 4: Make Predictions and Evaluate")  # Subheader for predictions and evaluation
st.write("""
The trained model predicts target values (y) for the testing set (X_test). These predictions are compared 
to the actual values (y_test) to assess the model's accuracy.
""")  # Explain the evaluation process

y_pred = model.predict(X_test)  # Make predictions on the testing set

# Plot predictions vs actual values
fig, ax = plt.subplots()  # Create a figure and axes for the plot
ax.scatter(y_test, y_pred, color='red', alpha=0.7)  # Scatter plot of actual vs predicted values
ax.set(title="Predicted vs Actual Values", xlabel="Actual Values", ylabel="Predicted Values")  # Set plot title and axis labels
st.pyplot(fig)  # Display the plot in the Streamlit app

# Model Performance Metrics
st.write("Model Performance Metrics:")  # Subsection for performance metrics
st.write("""
- **Mean Squared Error (MSE):** Measures the average squared difference between predicted values and actual values. 
  A lower MSE indicates better model performance.
- **Root Mean Squared Error (RMSE):** The square root of MSE, providing the error in the same units as the target variable. 
  RMSE is often easier to interpret and compare for real-world applications.
- **R-squared (R²):** Indicates how well the model explains the variability of the target variable. An R² value 
  close to 1 means the model fits the data well.

For a detailed comparison between RMSE (Root Mean Squared Error) and R-squared, check out this resource: 
[RMSE vs. R-squared](https://www.statology.org/rmse-vs-r-squared/)
""")  # Explain each metric and provide a resource link for further reading

# Calculate metrics
mse = np.mean((y_test - y_pred) ** 2)  # Calculate MSE
rmse = np.sqrt(mse)  # Calculate RMSE as the square root of MSE
r2 = model.score(X_test, y_test)  # Calculate R² score

# Display metrics
st.write(f"Mean Squared Error (MSE): {mse:.2f}")  # Display MSE
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")  # Display RMSE
st.write(f"R-squared (R²): {r2:.2f}")  # Display R² score

st.write("""
Congratulations! You've completed the tutorial. By understanding how linear regression models a relationship, 
you can apply it to real-world problems, such as predicting house prices, stock trends, and more.
""")  # Closing remarks for the tutorial