Machine Learning App Project
=================
This project dives into machine learning by comparing linear and logistic regression. The app will demonstrate their distinct purposes and applications in predictive modeling.

Project Overview
----------------
The goal of this project is to understand the differences, practical insight, and real world application of linear and logistic regression by providing hands-on experimentation with datasets, the app allows users to explore machine learning.

**What is Machine Learning?**

Machine learning is when humans give the computer the input and the output for the computer to learn the rules and connext them.

**Key Idea:** We use data to create models that make predictions or decisions.
- Remember: Gather and inspect data.
- Formulate: Build a model.
- Predict: Use the model to forecast new outcomes.

Linear Regression
-------------------
**When to Use It:**
- Target is numeric.
- Relationship is roughly linear.
- Data is not heavily influenced by outliers.

**Type of Data:**
- Features: Typically numeric or encoded categorical.
- Target: Continuous Numeric (e.g., price, test score)

**Questions It Answers:**
- “How can we predict a continuous outcome based on known inputs?”
- “What is the relationship between these features and the target?”

Logistic Regression
-------------------
**When to Use It:**
- Target: Binary Category (Yes/No, Pass/Fail, 1/0)
- Relationship: Features influence the probability of outcome ((log-)odds of event happening).

**Type of Data:**
Features: Typically numeric or encoded categorical.
Target: Always categorical

**Questions It Answers:**
- “What factors significantly impact the probability of a specific outcome?”
- “How likely is a certain event or class to occur?”

Instructions
------------
Step-by-step instructions on how to run the app locally and a link to the deployed version:
1. Import the pandas library in Python and assign it as pd
2. Import the pyplot functions in the matplotlib package and assign it as plt
3. Import the seaborn visualization library and assign it as sns
4. Load the CSV file of the dataset using the relative path for the CSV
5. Refer to the [Data Cleaning Codebook](Data_Cleaning_Visualization.ipynb) for the entire notebook.

References
----------
Links for further regression information!

- [What is Machine Learning?](file:///Users/justinsapienza/Downloads/Grokking%20ML_Ch%201%20&%202a.pdf)
- [Linear Regression](https://vita.had.co.nz/papers/tidy-data.pdf](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.fit))
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
