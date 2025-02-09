import streamlit as st
import pandas as pd

# streamlit run GitHub/SAPIENZA-Data-Science-Portfolio/basic-streamlit-app/main.py

st.title("Palmer's Penguins")

st.write("The Streamlit App creates widgets to visualize the Palmer's Penguins data in an interactive display.")

# Importing the dataset
penguins = pd.read_csv("GitHub/SAPIENZA-Data-Science-Portfolio/basic-streamlit-app/data/penguins.csv")

# Displaying the table in Streamlit
st.write("Here's an interactive table.")
st.dataframe(penguins)

# Using a selectbox to allow users to filter data by species
species = st.selectbox("Select a Species:", penguins["species"].unique())
# Filtering the DataFrame based on user selection
filtered_df = penguins[penguins["species"] == species]
# Display the filtered results
st.dataframe(filtered_df)

# Using a slider to allow users to filter data by body mass range
body_mass = st.slider("Choose a body mass range:",
                   min_value = penguins["body_mass_g"].min(),
                   max_value = penguins["body_mass_g"].max())
# Displaying the slider and filtered data in Streamlit
st.write(f"Penguins under {body_mass}:")
st.dataframe(penguins[penguins['body_mass_g'] <= body_mass])