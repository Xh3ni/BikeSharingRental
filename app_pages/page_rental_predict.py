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

    bike = load_data()

    rental_pipe = load_pkl_file(
        f'./inputs/datasets/raw/bike-share.pkl'
    )

    st.write('#### Rental Prediction')
    st.info(
        f"* The pipeline is composed of the transformations and the algorithm used to train the model.\n"
        f"* The pipeline performance on train based on the same statistical distributions and category encodings used with the training data."
    )


    X = bike[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit','temp', 'hum', 'windspeed']].values
    y = bike['cnt'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 0)

    # Define preprocessing for numeric columns (scale them)
    numeric_features = [6,7,8,9]
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Define preprocessing for categorical features (encode them)
    categorical_features = [0,1,2,3,4,5]
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create preprocessing and training pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', GradientBoostingRegressor())])


    # fit the pipeline to train a linear regression model on the training set
    model = pipeline.fit(X_train, (y_train))
    st.code(model)

    st.info(
        f"* Let's see how it performs with the validation data."
    )

    # Get predictions
    predictions = model.predict(X_test)

    # Display metrics
    mse = mean_squared_error(y_test, predictions)
    st.write("MSE:", mse)
    rmse = np.sqrt(mse)
    st.write("RMSE:", rmse)
    r2 = r2_score(y_test, predictions)
    st.write("R2:", r2)

    # Plot predicted vs actual
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Daily Bike Share Predictions')
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test,p(y_test), color='magenta')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


    rental_bike_pipline = load_pkl_file('./inputs/datasets/raw/bike-share.pkl')

    st.write('---')

    # Try an alternative algorithm

    st.info(
        f"* Using an alternative algorithm."
    )

    # Use a different estimator in the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor())])


    # fit the pipeline to train a linear regression model on the training set
    model = pipeline.fit(X_train, (y_train))
    st.code(model, "\n")

    # Get predictions
    predictions = model.predict(X_test)

    # Display metrics
    mse = mean_squared_error(y_test, predictions)
    st.write("MSE:", mse)
    rmse = np.sqrt(mse)
    st.write("RMSE:", rmse)
    r2 = r2_score(y_test, predictions)
    st.write("R2:", r2)

    # Plot predicted vs actual
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Daily Bike Share Predictions - Preprocessed')
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test,p(y_test), color='magenta')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.write('---')

    if st.checkbox("Use Trained Model"):
        loaded_model = rental_pipe

        # Create a numpy array containing a new observation (for example tomorrow's seasonal and weather forecast information)
        X_new = np.array([[1,1,0,3,1,1,0.226957,0.22927,0.436957,0.1869]]).astype('float64')
        st.code('New sample: {}'.format(list(X_new[0])))

        # Use the model to predict tomorrow's rentals
        result = loaded_model.predict(X_new)
        st.code('Prediction: {:.0f} rentals'.format(np.round(result[0])))
        st.write('---')
        st.info(
            f"* Suppose you have a weather forecast for the next five days;" 
            f" you could use the model to predict bike rentals for each day based on the expected weather conditions."
        )

        # An array of features based on five-day weather forecast
        X_new = np.array([[0,1,1,0,0,1,0.344167,0.363625,0.805833,0.160446],
                            [0,1,0,1,0,1,0.363478,0.353739,0.696087,0.248539],
                            [0,1,0,2,0,1,0.196364,0.189405,0.437273,0.248309],
                            [0,1,0,3,0,1,0.2,0.212122,0.590435,0.160296],
                            [0,1,0,4,0,1,0.226957,0.22927,0.436957,0.1869]])

        # Use the model to predict rentals
        results = loaded_model.predict(X_new)
        st.write('5-day rental predictions:')
        for prediction in results:
            st.code(np.round(prediction))




 
