import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# App title
st.title("Formula 1 Encyclopedia")

# App description
st.markdown("""
This app provides a simple interface to explore Formula 1 data (e.g. drivers, teams, circuits, races, etc.).
* Developed with Python, Streamlit, Pandas, etc.
* Driver Stats retrieved from the following [API](https://documenter.getpostman.com/view/11586746/SztEa7bL).
""")

# App filters
st.sidebar.header("Filters")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2022))))

# Web scraping of Formula 1 driver stats
# API documentation: https://documenter.getpostman.com/view/11586746/SztEa7bL
@st.cache
def load_data(year: int):
    url = f"http://ergast.com/api/f1/{year}/drivers.json?limit=100"
    json = pd.read_json(url)["MRData"]["DriverTable"]["Drivers"]
    df = pd.json_normalize(json)\
            .fillna("?")\
            .drop(["driverId", "url"], 1)\
            .reindex(columns=["code", "givenName", "familyName", "permanentNumber", "dateOfBirth", "nationality"])\
            .rename(columns={
                "givenName": "First Name",
                "familyName": "Last Name",
                "permanentNumber": "Number",
                "code": "Abbreviation",
                "dateOfBirth": "Date of Birth",
                "nationality": "Nationality"
            })

    return df

driver_stats = load_data(selected_year)

# Data display
st.header(f"Driver stats for the year {selected_year}")
st.write(f"Data Dimension: {str(driver_stats.shape[0])} rows and {str(driver_stats.shape[1])} columns.")
st.dataframe(driver_stats)

# Download CSV file
def df_to_csv(df):
    csv = df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64, {b64}" download="driver_stats.csv">Download CSV File</a>'
    return href

st.markdown(df_to_csv(driver_stats), unsafe_allow_html = True)
