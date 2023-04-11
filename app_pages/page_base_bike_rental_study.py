import streamlit as st
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_management import load_data


def page_base_bike_rental_study_body():

    # load data
    bike = load_data()

    vars_to_study = ['Temperature', 'Humidity', 'Windspeed', 'Rental']

    st.write("### Bike Rental Study")
    st.info(
        f"* The client is interested in having a study that visually "
        f"the usage of rental bikes on different seasons, months, hours, weeks, etc.")

    # inspect data
    if st.checkbox("Inspect Base Rental Usage"):
        st.write(
            f"* The dataset has {bike.shape[0]} rows and {bike.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(bike.head(10))

    st.write("---")

    # Bike Usage per Week
    st.write(
        f"* Print out bike usage per Week"
    )

    def CtnWeekly():
        bike['cnt'].asfreq('W').plot(linewidth=3)
        plt.title('Bike Usage Per week')
        plt.xlabel('Week')
        plt.ylabel('Bike Rental')

    CtnWeekly()

    if st.checkbox("Rental Weekly"):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    st.write("---")

    # Bike Usage per Month
    st.write(
        f"* Print out bike usage per Month"
    )

    if st.checkbox("Monthly Usage"):
        bike['cnt'].asfreq('M').plot(linewidth=3)
        plt.title('Bike Usage Per Month')
        plt.xlabel('Month')
        plt.ylabel('Bike Rental')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to bike Rental Usage. \n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    # Text based on "03. Visualize Dataset" notebook - "Visualise the entire Data" section
    st.info(
        f" Here is a clearer idea of the distribution of rentals values by visualizing the data. "
    )


    X_numerical = bike[['temp', 'hum', 'windspeed', 'cnt']]

    if st.checkbox("Rental Usage Distribution"):
        def show_distribution(var_data):
            fig, ax = plt.subplots(1, 2, figsize=(8, 8))

            ax[0].hist(var_data, bins=100)
            ax[0].set_xlabel('Frequency')

            mean_val = var_data.mean()
            median_val = var_data.median()
            min_val = var_data.min()
            max_val = var_data.max()
            mode_val = var_data.mode()[0]

            ax[0].axvline(mean_val, color='magenta', linestyle='dashed', linewidth=2)
            ax[0].axvline(median_val, color='black', linestyle='dashed', linewidth=2)

            ax[1].boxplot(var_data, vert=False)
            ax[1].set_xlabel('value')

            fig.suptitle(var_data.name)

            st.pyplot(fig=fig)
    
        for col in X_numerical:
            show_distribution(X_numerical[col])
    

    # Visualising Catagorical Variables
    # Text based on "03. Visualize Dataset" notebook - "Visualising Catagorical Variables" section
    st.info(
        f" Here is a clearer idea of the distribution of rentals values by visualizing Categorical data. "
    )

    X_cat = bike[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]

    if st.checkbox('Visualize Check categorical variables frequency'):
        for col in X_cat:
            fig = plt.figure(figsize=(9, 6))
            ax = fig.gca()
            cat_count = bike[col].value_counts().sort_index()
            cat_count.plot.bar(x=col,y='Rentals')
            ax.set_title(col + ' counts')
            ax.set_xlabel(col)
            ax.set_ylabel('Rentals')
            st.pyplot(fig=fig)
    st.write("---")



    
    

