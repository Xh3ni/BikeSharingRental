import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_base_bike_rental_study import page_base_bike_rental_study_body
from app_pages.page_rental_correlation import page_bike_rental_correlation_body
from app_pages.page_rental_predict import page_rental_predict_body
from app_pages.page_rental_predict_pipeline import page_rental_predict_pipeline_body

app = MultiPage(app_name= "Bike Rental") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Bike Rental Study", page_base_bike_rental_study_body)
app.add_page("Bike Rental Correlation", page_bike_rental_correlation_body)
app.add_page("Bike Rental Predictions", page_rental_predict_body)
app.add_page("Bike Rental Pipeline", page_rental_predict_pipeline_body)


app.run() # Run the  app