import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"The Project is both a data analysis and a machine learning application.\n"
        f"The data analysis helps users understand what factors affect the use of bike "
        f"insted of other transport services such as bus or subway, the duration of travel, "
        f"departure and arrival position.\n"
        f"The machine learning application allows users to input temperature, humidity, wind speed,...etc "
        f"information and get predictions of bike rental usage.\n\n "
        f"**Project Dataset**\n"
        f"* The dataset contains the hourly and daily count of rental bikes "
        f"between years 2011 and 2012 in Capital bikeshare system in Washington, DC "
        f"with the corresponding weather and seasonal information.")

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Xh3ni/BikeSharingRental).")
    

    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - Rental Rates: The rental rates should be competitive and adjusted based on the season, "
        f" month, hour, and weather condition.\n\n"
        f" 2 - Rental Duration: The rental duration should be flexible and aligned with the workingday "
        f"and holiday schedule.\n\n"
        f" 3 -  Rental History and Analytics\n\n"
        f"* Keep track of user rental history and generate usage reports for analytical purposes\n\n"
        f"* Use collected data (e.g., weather, time, location) to optimize bike availability, pricing, and promotions\n\n"
        f"4 - Weather and Environmental Considerations\n\n"
        f"* Adjust bike availability and rental recommendations based on current and forecasted weather conditions\n\n"
        f"* Encourage environmentally friendly practices by promoting bike usage during low-emission periods\n\n"
        )