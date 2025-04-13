import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Title of the App
st.title("Logistic Regression App")
st.write("""
Upload your own dataset or select a real-world dataset (e.g., Iris or Titanic) to perform logistic regression and analyze results.
""")

# Section 1: Dataset Selection
st.header("Step 1: Select or Upload a Dataset")

# Real-world Sample Datasets
sample_datasets = {
    "Iris Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Titanic Dataset": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
}

# Dropdown for Sample Dataset Selection
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

# Section 2: Logistic Regression
if data is not None:
    st.header("Step 2: Perform Logistic Regression")

    # Feature and Target Selection
    features = st.multiselect("Select Features (Independent Variables)", options=data.columns)
    target = st.selectbox("Select Target Variable (Dependent Variable)", options=data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Handle non-numeric target variables
        if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
            st.write("Target variable contains categorical values. Encoding them...")
            y = pd.factorize(y)[0]  # Convert categories to numeric values

        # Train-Test Split
        test_size = st.slider("Test Size (Percentage)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train Logistic Regression Model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Display Model Performance Metrics
        st.subheader("Model Performance Metrics")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)

        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"AUC Score: {auc_score:.2f}")

        # ROC Curve Visualization
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

        st.write("Experiment with different features or adjust the test size to see how the model performs!")
    else:
        st.info("Please select at least one feature and a target variable.")
else:
    st.info("Please select a sample dataset or upload your own dataset to proceed.")