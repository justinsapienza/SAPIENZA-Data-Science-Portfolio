# Import necessary libraries
import streamlit as st  # Streamlit for building interactive web apps
import pandas as pd  # Pandas for data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.linear_model import LogisticRegression  # For performing logistic regression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve  # Evaluation metrics
import matplotlib.pyplot as plt  # For visualizing the ROC curve

# Title of the App
st.title("Logistic Regression App")  # Set the app title
st.write("""
Upload your own dataset or select a real-world dataset (e.g., Iris or Titanic) to perform logistic regression and analyze results.
""")  # Description and instructions for the app

# Section 1: Dataset Selection
st.header("Step 1: Select or Upload a Dataset")  # Header for dataset selection step

# Real-world Sample Datasets
sample_datasets = {
    "Titanic Dataset": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"  # URL for Titanic dataset
}

# Dropdown for Sample Dataset Selection
selected_sample = st.selectbox("Select a Sample Dataset", ["None"] + list(sample_datasets.keys()))  # User selects a sample dataset from the dropdown

data = None  # Initialize a placeholder for the dataset

if selected_sample != "None":
    # Load the selected dataset
    dataset_url = sample_datasets[selected_sample]  # Get the URL of the selected dataset
    data = pd.read_csv(dataset_url)  # Load the dataset into a DataFrame
    st.write(f"Preview of {selected_sample}:")  # Display the dataset name
    st.dataframe(data)  # Show a preview of the dataset

# File uploader for custom dataset
uploaded_file = st.file_uploader("Or Upload Your Own Dataset", type="csv")  # Allow the user to upload their own dataset

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)  # Load the uploaded dataset into a DataFrame
    st.write("Preview of Your Uploaded Dataset:")  # Notify the user that their dataset is loaded
    st.dataframe(data)  # Show a preview of the uploaded dataset

# Section 2: Logistic Regression
if data is not None:  # Check if a dataset is available
    st.header("Step 2: Perform Logistic Regression")  # Header for logistic regression step

    # Handle missing values
    st.write("Checking for missing values...")  # Notify the user about missing value handling
    data = data.dropna()  # Drop rows with missing values

    # Feature and Target Selection
    features = st.multiselect("Select Features (Independent Variables)", options=data.columns)  # User selects features
    target = st.selectbox("Select Target Variable (Dependent Variable)", options=data.columns)  # User selects the target variable

    if features and target:  # Proceed if both features and a target are selected
        # Display a warning about the target variable
        st.warning(
            "Please ensure the target variable is binary (e.g., 0 and 1). "
            "For the Titanic dataset, the 'Survived' column is suitable as it indicates survival (1) or non-survival (0). "
            "Logistic regression requires a target variable with discrete classes. "
            "If your target variable is continuous, consider binning it into categories."
        )  # Provide guidelines for target variable selection

        X = data[features]  # Extract the selected features
        y = data[target]  # Extract the selected target variable

        # Handle non-numeric features
        st.write("Checking for non-numeric features...")  # Notify the user about encoding non-numeric features
        non_numeric_features = X.select_dtypes(include=["object", "category"]).columns  # Identify non-numeric columns
        if len(non_numeric_features) > 0:
            st.write(f"Encoding non-numeric features: {', '.join(non_numeric_features)}")  # List features being encoded
            X = pd.get_dummies(X, columns=non_numeric_features)  # Convert categorical features to numeric using one-hot encoding

        # Handle non-numeric or continuous target variable
        if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):  # Check if the target is categorical
            st.write("Target variable contains categorical values. Encoding them...")  # Notify user
            y = pd.factorize(y)[0]  # Encode target variable to numeric
        elif y.dtype in ["int64", "float64"]:  # Check if the target is continuous
            st.write("Target variable is continuous. Binning the values into discrete classes...")  # Notify user
            y = pd.qcut(y, q=2, labels=[0, 1], duplicates="drop")  # Bin continuous target into two categories

        # Train-Test Split
        test_size = st.slider("Test Size (Percentage)", 10, 50, 20) / 100  # Slider to select test set percentage
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)  # Split data into training and testing sets

        # Train Logistic Regression Model
        model = LogisticRegression()  # Initialize the logistic regression model
        model.fit(X_train, y_train)  # Train the model

        # Predictions
        y_pred = model.predict(X_test)  # Predict labels (0 or 1) on the testing set
        y_prob = model.predict_proba(X_test)[:, 1]  # Predict probabilities for the positive class (1)

        # Display Model Performance Metrics
        st.subheader("Model Performance Metrics")  # Header for performance metrics
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
        precision = precision_score(y_test, y_pred)  # Calculate precision
        recall = recall_score(y_test, y_pred)  # Calculate recall
        auc_score = roc_auc_score(y_test, y_prob)  # Calculate AUC score

        # Display metrics
        st.write(f"Accuracy: {accuracy:.2f}")  # Display accuracy
        st.write(f"Precision: {precision:.2f}")  # Display precision
        st.write(f"Recall: {recall:.2f}")  # Display recall
        st.write(f"AUC Score: {auc_score:.2f}")  # Display AUC score

        # ROC Curve Visualization
        st.subheader("ROC Curve")  # Header for ROC curve visualization
        fpr, tpr, _ = roc_curve(y_test, y_prob)  # Calculate FPR and TPR for the ROC curve

        plt.figure(figsize=(6, 4))  # Set figure size
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color="blue")  # Plot ROC curve
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Plot diagonal reference line
        plt.xlabel("False Positive Rate")  # X-axis label
        plt.ylabel("True Positive Rate")  # Y-axis label
        plt.title("Receiver Operating Characteristic (ROC) Curve")  # Title of the plot
        plt.legend(loc="lower right")  # Position the legend
        st.pyplot(plt)  # Display the plot

        st.write("Experiment with different features or adjust the test size to see how the model performs!")  # Encourage users to explore
    else:
        st.info("Please select at least one feature and a target variable.")  # Prompt users to select variables
else:
    st.info("Please select a sample dataset or upload your own dataset to proceed.")  # Prompt users to upload/select a dataset