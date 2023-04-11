import streamlit as st

def predict_rental(X_live, rental_features, rental_pipeline, rental_pipeline_model):

    # from live data, subset features related to this pipeline

    X_live_rental = X_live.filter(rental_features)

    # apply data cleaning / feat engine pipeline to live data
    X_live_rental_dc = rental_pipeline.transform(X_live_rental)

    # predict

    rental_prediction = rental_pipeline_model.predict(X_live_rental_dc)
    rental_prediction_prob = rental_pipeline_model.predict_prob(X_live_rental_dc)
    # st.write(rental_prediction_prob)

   



