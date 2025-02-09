import streamlit as st
import pandas as pd

# streamlit run GitHub/SAPIENZA-Data-Science-Portfolio/basic-streamlit-app/main.py

st.title("Palmer's Penguins")

st.write("The Streamlit App creates widgets to visualize the Palmer's Penguins data in an interactive display.")

penguins = pd.read_csv("GitHub/SAPIENZA-Data-Science-Portfolio/basic-streamlit-app/data/penguins.csv")

st.write("Here's an interactive table.")
st.dataframe(penguins)

species = st.selectbox("Select a Species:", penguins["species"].unique())
filtered_df = penguins[penguins["species"] == species]
st.dataframe(filtered_df)

body_mass = st.slider("Choose a body mass range:",
                   min_value = penguins["body_mass_g"].min(),
                   max_value = penguins["body_mass_g"].max())
st.write(f"Penguins under {body_mass}:")
st.dataframe(penguins[penguins['body_mass_g'] <= body_mass])