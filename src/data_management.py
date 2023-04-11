import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data():
    bike = pd.read_csv('inputs/datasets/raw/bike_sharing_daily.csv')
    bike = bike.drop(labels=['instant'], axis=1)
    bike = bike.drop(labels=['casual', 'registered'], axis=1)
    bike.dteday = pd.to_datetime(bike.dteday, format='%m/%d/%Y')
    bike.index = pd.DatetimeIndex(bike.dteday)
    bike = bike.drop(labels=['dteday'], axis=1)
    return bike


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)