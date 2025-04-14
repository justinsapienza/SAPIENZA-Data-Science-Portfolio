# Import the Streamlit Library
import streamlit as st

# Create Page Layout
st.set_page_config(page_title = "Home", layout = "centered")
st.title("Supervised Machine Learning Tutorial")

# Use Markdowns For App Overview
st.markdown(
    """
    Welcome to the Supervised Machine Learning Tutoral. This web app is a versatile tool that compares **linear regression** and **logistic regression**. Whether you are interested in linear regression or logistic regression, this app dives into the distinctions between the two powerful tools in predictive modeling. The models in this web app are custom-built to provide and interactive platform for exploring how these models work, comparing their outputs, and understanding their applications.

    ## Key Features

    ### Linear Regression
    Linear regression is a statistical method used to model the relationship between one or more independent variables (features) and a dependent variable (target). It assumes that this relationship is linear, meaning it can be represented as a straight line. In simple linear  regression, the model predicts the dependent variable using a single indpendent variable, whereas in multiple linear regression, multiple features are used. The model aims to estimate the coefficients of the line that minimizes the difference between actual and predicted values. Linear regression is applied in forecasting, such as predicting house prices based on square footage or sales revenue based on marketing expenditures. Overall, its simplicity and interpretability make it a fundamental tool in machine learning.

    ### Logistic Regression
    Logistic regression is a statistical modeling technique used for binary classification tasks, where the goal is to predict one of two possible outcomes (e.g., yes/no, true/false). Unlike linear regression, logistic regression does not predict continuous values but instead estimates the probability that a given observation belongs to a particular class. The model is particularly useful when the relationship between independent and the dependent variable is non-linear but can be expressed through logarthmic odds. Logistic regression is applied in real-world scenarios, such as diagnosing diseases and identifying fraud, due to its simplicity, intepretability, and effectiveness in handling classification problems.

    ## Directions

    1. Utilize the arrow in the top left corner to toggle between the tutorials for linear and logistic regression.

    2. Follow the instructions on the linear and logistic regression analysis tabs to experiment with the models.

    3. Explore linear and logistic regression and have fun!
""",
)
