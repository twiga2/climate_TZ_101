#loading the data and pre-processing the data
#Import libraries
import pandas as pd 
import numpy as np
import streamlit as st


# Load the data
@st.cache_data
def load_data():
    """
    Load the data for climate from the csv file
    """
    climatedf = pd.read_csv("data/tanzania_climate_data.csv")
    
    return climatedf

def prepare_features(climatedf):
    """
    Prepare the features for model training
    """
    # Cyclical encoding for months (helps model understand seasonality)
    climatedf['month_sin'] = np.sin(2 * np.pi * climatedf['Month'] / 12)
    climatedf['month_cos'] = np.cos(2 * np.pi * climatedf['Month'] / 12)

    #lag features for time series forecasting 
   # climatedf['lag_1'] = climatedf['Average_Temperature_C'].shift(1) # previous month
   # climatedf['lag_12'] = climatedf['Average_Temperature_C'].shift(12) #same month last year


    # dropped other lag_1, lag_12, Total_Rainfall_mm, Year, Month due to heatmap results
    X = climatedf[['Max_Temperature_C','Min_Temperature_C','month_sin','month_cos']].values
    y = climatedf['Average_Temperature_C'].values

    return X, y