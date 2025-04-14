# Import necessary libraries
import streamlit as st  # Streamlit for building interactive web applications
import pandas as pd  # Pandas for data manipulation and analysis
import numpy as np  # NumPy for numerical computations
from sklearn.model_selection import train_test_split  # Split data into training and testing sets
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.metrics import mean_squared_error, r2_score  # Metrics for evaluating model performance
import matplotlib.pyplot as plt  # Matplotlib for data visualization

# Page Title
st.title("Linear Regression Analysis")  # Set the title for the app
st.write("""
Upload your own dataset or select a real-world dataset (e.g., Iris dataset) to perform linear regression and analyze results.
""")  # Provide an introduction to the app

# Section 1: Dataset Selection
st.header("Step 1: Select or Upload a Dataset")  # Add a header for the first step

# Real-world Sample Dataset
sample_datasets = {
    "Iris Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",  # URL for Iris dataset
    "Palmer's Penguins Dataset": "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"  # URL for Penguins dataset
}

selected_sample = st.selectbox("Select a Sample Dataset", ["None"] + list(sample_datasets.keys()))  # Dropdown for selecting a dataset

data = None  # Initialize the dataset variable

if selected_sample != "None":
    # Load the selected dataset
    dataset_url = sample_datasets[selected_sample]  # Get URL of the selected dataset
    data = pd.read_csv(dataset_url)  # Read the dataset into a pandas DataFrame
    st.write(f"Preview of {selected_sample}:")  # Display the name of the selected dataset
    st.dataframe(data)  # Show a preview of the dataset

# File uploader for custom dataset
uploaded_file = st.file_uploader("Or Upload Your Own Dataset", type="csv")  # Allow users to upload their own dataset

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)  # Load the uploaded dataset into a pandas DataFrame
    st.write("Preview of Your Uploaded Dataset:")  # Inform the user about their uploaded dataset
    st.dataframe(data)  # Display a preview of the dataset

# Section 2: Linear Regression
if data is not None:  # Check if a dataset is available
    st.header("Step 2: Perform Linear Regression")  # Add a header for the second step

    # Handle missing values
    if data.isnull().values.any():  # Check if there are missing values
        st.write("Dataset contains missing values. Filling missing values with column medians...")  # Notify the user about missing values
        data = data.fillna(data.median(numeric_only=True))  # Fill missing values in numeric columns with column medians
        for col in data.select_dtypes(include=["object", "category"]).columns:
            data[col] = data[col].fillna(data[col].mode()[0])  # Fill missing values in categorical columns with the mode

    # Feature and Target Selection
    features = st.multiselect("Select Features (Independent Variables)", options=data.columns)  # Allow users to select independent variables (features)
    target = st.selectbox("Select Target Variable (Dependent Variable)", options=data.columns)  # Allow users to select the dependent variable (target)

    if features and target:  # Check if both features and target are selected
        # Display a warning about the target variable
        st.warning(
            "Please ensure the target variable is continuous for linear regression. "
            "Linear regression predicts continuous outcomes, such as numerical values."
        )

        X = data[features]  # Extract selected features
        y = data[target]  # Extract the target variable

        # Handle non-numeric features
        non_numeric_cols = X.select_dtypes(include=["object", "category"]).columns  # Identify non-numeric feature columns
        if not non_numeric_cols.empty:
            st.write("Features contain non-numeric values. Encoding them...")  # Notify the user about encoding
            X = pd.get_dummies(X, columns=non_numeric_cols)  # Encode categorical features using one-hot encoding

        # Handle non-numeric target variable
        if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):  # Check if the target is categorical
            st.write("Target variable contains categorical values. Encoding them...")  # Notify the user about encoding
            y = pd.factorize(y)[0]  # Encode the target variable into numeric values

        # Train-Test Split
        test_size = st.slider("Test Size (Percentage)", 10, 50, 20) / 100  # Slider for selecting test size percentage
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)  # Split the dataset into training and testing sets

        # Train Linear Regression Model
        model = LinearRegression()  # Initialize the Linear Regression model
        model.fit(X_train, y_train)  # Train the model using the training data

        # Predictions
        y_pred = model.predict(X_test)  # Make predictions on the testing set

        # Display Model Coefficients
        st.subheader("Model Coefficients")  # Add a subheader for displaying coefficients
        coefficients = pd.DataFrame({
            "Feature": X_train.columns,  # List of feature names
            "Coefficient": model.coef_  # Corresponding coefficients
        })
        st.dataframe(coefficients)  # Display the coefficients in a table

        # Display Model Performance Metrics
        st.subheader("Model Performance Metrics")  # Add a subheader for performance metrics

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
        rmse = np.sqrt(mse)  # Calculate Root Mean Squared Error
        r2 = r2_score(y_test, y_pred)  # Calculate R-squared score

        # Display metrics
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")  # Display MSE
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")  # Display RMSE
        st.write(f"R-squared (R²): {r2:.2f}")  # Display R-squared score

        # Visualize Predictions vs Actual Values
        st.subheader("Predicted vs Actual Values")  # Add a subheader for the visualization
        fig, ax = plt.subplots()  # Create a figure and axes for the plot
        ax.scatter(y_test, y_pred, color='blue', alpha=0.7)  # Create a scatter plot of actual vs predicted values
        ax.set(title="Predicted vs Actual Values", xlabel="Actual Values", ylabel="Predicted Values")  # Set plot title and axis labels
        st.pyplot(fig)  # Display the plot in the Streamlit app

        st.write("Explore different feature combinations or adjust the test size to see how the model performs!")  # Encourage users to experiment
    else:
        st.info("Please select at least one feature and a target variable.")  # Notify users to select variables
else:
    st.info("Please select a sample dataset or upload your own dataset to proceed.")  # Notify users to provide a dataset