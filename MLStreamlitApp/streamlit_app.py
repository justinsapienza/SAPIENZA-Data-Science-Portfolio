# Import the Streamlit Library
import streamlit as st

# Create Page Layout
st.set_page_config(page_title = "Home", layout = "centered")
st.title("Supervised Machine Learning Tutorial")

# Use Markdowns For App Overview
st.markdown(
    """
    Welcome to the Supervised Machine Learning Tutoral. This web app is a versatile tool that compares linear and logistic regression. Whether you are interested in linear regression or logistic regression, this app dives into the distinctions between the two powerful tools in predictive modeling. The models in this web app are custom-built to provide and interactive platform for exploring how these models work, comparing their outputs, and understanding their applications.

    ## Key Features

    ### Linear Regression
    Linear regression predicts continuous outcomes by modeling the relationship between independent variables and a numeric dependent variable.

    ### Logistic Regression
    Logistic regression is used for classification tasks to predict the probability of a categorical outcome. It models the relationship between independent varaibles and the dependent variable.

""",
)
