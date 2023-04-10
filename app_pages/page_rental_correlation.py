import streamlit as st
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyter_notebooks import *


def page_bike_rental_correlation_body():

    # Correlation Study Summary

    # load data
    bike = pd.read_csv('inputs/datasets/raw/bike_sharing_daily.csv')
    bike = bike.drop(labels=['instant'], axis=1)
    bike = bike.drop(labels=['casual', 'registered'], axis=1)
    bike.dteday = pd.to_datetime(bike.dteday, format='%m/%d/%Y')
    bike.index = pd.DatetimeIndex(bike.dteday)
    bike = bike.drop(labels=['dteday'], axis=1)
    bike

    vars_to_study = ['Temperature', 'Humidity', 'Windspeed', 'Rental']

    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to bike Rental Usage. \n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    # Text based on "03. Visualize Dataset" notebook - "Visualise the entire Data" section
    st.info(
        f" Find correlation between numerical variables with label using scatter charts. "
    )

    X_numerical = bike[['temp', 'hum', 'windspeed', 'cnt']]

    if st.checkbox("Use Correlation"):
        for col in X_numerical:
            correlation_value = bike[col].corr(bike['cnt'])
            fig = plt.figure(figsize=(9, 6))
            plt.scatter(x=bike[col],y=bike['cnt'], color='steelblue')
            plt.title("correlation_value: " + str(correlation_value))
            plt.xlabel(col) 
            plt.ylabel("Rentals")
            st.pyplot(fig=fig)
    st.write("---")

    # Check the correlation between variables

    st.info(
        f"Check the Correlation between data."
    )

    if st.checkbox("Check Correlation"):
        sns.heatmap(X_numerical.corr(), annot =True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()