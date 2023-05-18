import streamlit as st
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.data_management import load_data, load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_rental_predict_body():

    # load needed file
    rental_pipe = load_pkl_file('./inputs/datasets/raw/bike-share.pkl')

    st.write('#### Rental Prediction')
    st.info(
        f"* Here the user can enter values for the features used in the model, and a button to predict rentals based on those values.\n"
        f"* When the user clicks the button, the model is used to make the prediction, and the predicted rentals are displayed on the screen."
    )

    # Create a form to enter feature values
    st.write('### Enter feature values')
    season = st.selectbox('Season', [1, 2, 3, 4])
    year = st.selectbox('Year', [0, 1])
    month = st.selectbox('Month', range(1, 13))
    holiday = st.selectbox('Holiday', [0, 1])
    weekday = st.selectbox('Weekday', range(0, 7))
    workingday = st.selectbox('Workingday', [0, 1])
    weathersit = st.selectbox('Weathersit', [1, 2, 3])
    temp = st.number_input('Temperature', value=0.0, step=0.1)
    hum = st.number_input('Humidity', value=0.0, step=0.1)
    windspeed = st.number_input('Windspeed', value=0.0, step=0.1)

    # Create a button to predict rentals
    if st.button('Predict Rentals'):
        # Use the model to predict rentals
        X_new = np.array([[season, year, month, holiday, weekday, workingday,
                         weathersit, temp, hum, windspeed]]).astype('float64')
        prediction = rental_pipe.predict(X_new)[0]

        # Display the predicted rentals
        st.success('Predicted rentals: {:.0f}'.format(prediction))
