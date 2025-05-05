
Basic Streamlit App
=====================================
A streamlit is a Python-based web app that allows users to interactively explore data, visualize results, and execute models without needing complex skills.

Project Overview
----------------
The goal of this project is to create a simple, interactive web application that transforms data analysis into an accessible and user-friendly experience. This involves showcasing key insights through user-friendly widgets like sliders and dropdowns. Using Python, Streamlit enables users to create data-driven applications using simple widgets.

Instructions
------------
Step-by-step instructions on how to run the app locally!
1. Create a python file
2. Import the streamlit library
```python
import streamlit as st
```
3. Right click on the file and select "Copy Relative Path"
4. Open the terminal in the bottom right corner
5. Type into the terminal "streamlit run" and paste the relative path
6. Hit enter
7. The app will open as a local host in your browser

App Features
------------
**Palmer's Penguins Dataset**
***
The Palmer's Penguins Dataset is a widely used dataset for data visualization and machine learning.

**Species Included:**
- Adelie Penguins (152 individuals)
- Gentoo Penguins (124 individuals)
- Chinstrap Penguins (68 individuals)
Total Instances: 344 Penguins

**Features in the Dataset:**
Each penguin is described by several attributes.
- Bill Length (mm) - Measurement of the penguin's beak length.
- Bill Depth (mm) – Thickness of the penguin’s beak.
- Flipper Length (mm) – Wing length, useful for species identification.
- Body Mass (g) – Weight of the penguin in grams.
- Sex – Male or female (some missing values).
- Island – The island where the penguin was observed (Biscoe, Dream, or Torgersen).

The Palmer's Penguins Dataset is ideal for teaching basic streamlit widgets and machine learning techniques that allow for enhanced data visualizations.

**Widgets**
***
**Sliders:** allows users to select a value within a specified range

Useful for adjusting parameters.

**Dropdowns:** allows users to choose an option from a predefined list

Useful for categorical inputs, model selections, and filtering data.

References
----------
See additional references that informed my app creation:

- [Creating a Streamlit App](https://docs.streamlit.io/get-started/tutorials/create-an-app)
- [Beginner's Guide to Streamlit](https://www.geeksforgeeks.org/a-beginners-guide-to-streamlit/)
