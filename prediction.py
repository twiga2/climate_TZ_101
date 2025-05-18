import numpy as np
import pandas as pd


#  Make prediction

def make_prediction(model, year, historical_avg):
    """
    Make prediction of temperature for a given month and year
    """

    #features = np.array([year, month])

    future_dates = pd.DataFrame({
        'Year': [year] *12,  # repeat the year for all months
        'Month': range(1, 13)  # Jan- Dec
    })

    #merge with historical averages
    forecast = future_dates.merge(historical_avg, on='Month')

    # Add cyclical month encoding
    forecast['month_sin'] = np.sin(2 * np.pi * forecast['Month'] / 12)
    forecast['month_cos'] = np.cos(2 * np.pi * forecast['Month'] / 12)

    X_forecast = forecast[[
        'Forecasted_Max_Temp',
        'Forecasted_Min_Temp',
        'month_sin', 
        'month_cos'
        ]].values
    
    #Predict Average Temperature
    forecast['Predicted_Avg_Temp'] = model.predict(X_forecast)
    
    return forecast[['Month', 'Forecasted_Min_Temp', 'Forecasted_Max_Temp', 'Predicted_Avg_Temp']]

# get historical contenxt
def get_historical_average(df):
    """
    Get the forecasted Min and Max temp using historical averages 
    """
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
    df = df.set_index('date')

    # Group by Month and calculate historical averages
    historical_avg = df.groupby('Month').agg({
    'Min_Temperature_C': 'mean',
    'Max_Temperature_C': 'mean'
        }).rename(columns={
    'Min_Temperature_C': 'Forecasted_Min_Temp',
    'Max_Temperature_C': 'Forecasted_Max_Temp'
        }).reset_index() 
    
    return historical_avg

def get_historical_context(df, month):
    """
    Get historical average temp for a given month
    """

    return df[df['month'] == month]['temperature'].mean()