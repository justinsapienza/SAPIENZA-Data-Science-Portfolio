Tidy Data Project
=================
This project will demonstrate the tidy data principles to clean, transform, and then do basic exploratory analysis on the data.
> "A huge amount of effort is spent cleaning data to get it ready for analysis, but there has been little research on how to make data cleaning as easy and effectiv as possible." - Hadley Wickman, Author, _Tiny Data_

Project Overview
----------------
The goal of this project is to transform and clean a messy dataset into a tidy format, aligning it with the principles outlined by Hadley Wickham in his _Tidy Data_ framework. By tidying a dataset, we aim to make it easier to analyze, visualize, and model the data efficiently.

**Key Principles of Tidy Data:**
1. Each variable forms its own column.
2. Each observation forms a row.
3. Each type of observational unit forms a table.

By achieving this format, the project seeks to unlock the full potential of the dataset, making it more accessible and usable for data analysis, visualization, and modeling processes.

Dataset Description
-------------------
The dataset for this project represents information about Olympic medalists in the 2008 Olympics.

**Columns (Variables)**
- medalist_name: names of the medalists
- Multiple columns that combine event and gender

**Rows (Observations)**
- Individual athletes, containing information about their name, events participated in, and medals they earned

**Potential Issues**

The original dataset is in a wide format. This means that columns are structured as combinations of variables. These columns need to be melted into a long format to separate variables cleanly.

Instructions
------------
Step-by-step instructions on how to run the notebook:
1. Import the pandas library in Python and assign it as pd
2. Import the pyplot functions in the matplotlib package and assign it as plt
3. Import the seaborn visualization library and assign it as sns
4. Load the CSV file of the dataset using the relative path for the CSV
5. Use the function print(df.head()) to preview the dataset.

References
----------
Links for further data tidying help!

- [_Tiny Data_ by Hadley Wickman](https://vita.had.co.nz/papers/tidy-data.pdf)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

Cleaned Data Visualizations
---------------------------


