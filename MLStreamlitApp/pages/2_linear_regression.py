import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Page Title
st.title("Linear Regression Analysis")
st.write("""
Upload your own dataset or select a real-world dataset (e.g., Iris dataset) to perform linear regression and analyze results.
""")

# Section 1: Dataset Selection
st.header("Step 1: Select or Upload a Dataset")

# Real-world Sample Dataset
sample_datasets = {
    "Iris Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Palmer's Penguins Dataset": "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
}

selected_sample = st.selectbox("Select a Sample Dataset", ["None"] + list(sample_datasets.keys()))

data = None

if selected_sample != "None":
    # Load the selected dataset
    dataset_url = sample_datasets[selected_sample]
    data = pd.read_csv(dataset_url)
    st.write(f"Preview of {selected_sample}:")
    st.dataframe(data)

# File uploader for custom dataset
uploaded_file = st.file_uploader("Or Upload Your Own Dataset", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Your Uploaded Dataset:")
    st.dataframe(data)

# Section 2: Linear Regression
if data is not None:
    st.header("Step 2: Perform Linear Regression")

    # Feature and Target Selection
    features = st.multiselect("Select Features (Independent Variables)", options=data.columns)
    target = st.selectbox("Select Target Variable (Dependent Variable)", options=data.columns)
    
    if features and target:
        X = data[features]
        y = data[target]

        # Handle non-numeric target variables
        if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
            st.write("Target variable contains categorical values. Encoding them...")
            y = pd.factorize(y)[0]  # Convert categories to numeric

        # Train-Test Split
        test_size = st.slider("Test Size (Percentage)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Display Model Coefficients
        st.subheader("Model Coefficients")
        coefficients = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
        st.dataframe(coefficients)

        # Display Model Performance Metrics
        st.subheader("Model Performance Metrics")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")

        # Visualize Predictions vs Actual Values
        st.subheader("Predicted vs Actual Values")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue', alpha=0.7)
        ax.set(title="Predicted vs Actual Values", xlabel="Actual Values", ylabel="Predicted Values")
        st.pyplot(fig)

        st.write("Explore different feature combinations or adjust the test size to see how the model performs!")
    else:
        st.info("Please select at least one feature and a target variable.")
else:
    st.info("Please select a sample dataset or upload your own dataset to proceed.")