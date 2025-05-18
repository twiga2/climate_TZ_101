#Import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_time_series(df):
    """
    Plot the temperatures and rainfall over time
    """
    # group by annual averages temperature and rainfall 

    annual_data = df.groupby("Year").mean().reset_index()

    fig, ax1 = plt.subplots(figsize=(14,8))

    # plot annual average temperature (left axis)
    ax1.plot(annual_data["Year"], annual_data["Average_Temperature_C"],
         'b', marker="o", label="Avg Temp")
    ax1.set_ylabel("Temperature (°C)", fontsize=14)
    ax1.set_xlabel("Year")
    ax1.tick_params(axis='y', labelcolor='b')

    # rainfall over the years (right axis)
    ax2 = ax1.twinx()
    ax2.plot(annual_data["Year"], annual_data["Total_Rainfall_mm"],
         'g', marker="o", label="Rainfall")
    ax2.set_ylabel("Rainfall (mm)", color='g', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='g')

    plt.title("Temperature vs Rainfall over the years (2000 to 2020)", fontsize=18)
    plt.grid(alpha=0.3)
    fig.legend(loc='upper left')
    return fig

def seasonal_temperature(df):
    """
    Identify seasonal patterns for temperature using decomposition techniques 
    """
    #creating the dataframe to datetime index
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
    df = df.set_index('date')

    # Make sure the data is sorted by date
    df = df.sort_index()
    climateseasonal=df.copy()
    climateseasonal = climateseasonal.drop(columns=['Year','Month'])

    # We analyze averege temperature)
    ts = climateseasonal['Average_Temperature_C']

    # Perform decomposition on the monthly data, use period=12 (12 months in a year)
    result = seasonal_decompose(ts, model='additive', period=12)
    # plot components individually 
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig1.suptitle("Seasonal patterns of monthly Average Temperature")
    result.observed.plot(ax=ax1, title='Observed')
    result.trend.plot(ax=ax2, title='Trend')
    result.seasonal.plot(ax=ax3, title='Seasonal')
    result.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()


    # Extract the seasonal component
    seasonal_component = result.seasonal

    # Get the average seasonal pattern by month
    monthly_seasonal = seasonal_component.groupby(seasonal_component.index.month).mean()

    # Plot the seasonal pattern
    fig2, ax5 = plt.subplots(figsize=(10, 4))
    # Plot the seasonal pattern using the axis object
    monthly_seasonal.plot(kind='bar', ax=ax5)
    ax5.set_title('Average Seasonal Pattern for Temperature by Month')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Seasonal Component')
    #Set x-ticks using the axis object
    ax5.set_xticks(ticks=range(12))
    ax5.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    #add grid using the axis object
    ax5.grid(alpha=0.3)
    plt.close('all')

    return fig1, fig2

def seasonal_rainfall(df):
    """
    Identify seasonal patterns for rainfall using decomposition techniques 
    """
    #creating the dataframe to datetime index
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
    df = df.set_index('date')

    # Make sure the data is sorted by date
    df = df.sort_index()
    climateseasonal=df.copy()
    climateseasonal = climateseasonal.drop(columns=['Year','Month'])

    # We analyze total rainfall
    ts2 = climateseasonal['Total_Rainfall_mm']

    # Perform decomposition on the monthly data, use period=12 (12 months in a year)
    result2 = seasonal_decompose(ts2, model='additive', period=12)
    # Plotting components individually 
    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig1.suptitle("Seasonal patterns of monthly Total Rainfall")
    result2.observed.plot(ax=ax1, title='Observed')
    result2.trend.plot(ax=ax2, title='Trend')
    result2.seasonal.plot(ax=ax3, title='Seasonal')
    result2.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()

    # Extract the seasonal component
    seasonal_component2 = result2.seasonal

    # Get the average seasonal pattern by month
    monthly_seasonal = seasonal_component2.groupby(seasonal_component2.index.month).mean()

    # Plot the seasonal pattern
    fig2, ax5 = plt.subplots(figsize=(10, 4))
    
    monthly_seasonal.plot(kind='bar', ax=ax5)
    ax5.set_title('Average Seasonal Pattern for Rainfall by Month')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Seasonal Component')
    ax5.set_xticks(ticks=range(12))
    ax5.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax5.grid(alpha=0.3)
    plt.close('all')
    return fig1, fig2

def heatmap_corr(df):
    """
    Heatmap indicating correlation between variables    
    """
    #correlation between variables

   # annual_data = df.groupby("Year").mean().reset_index()
    
    df['month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month']/12)

    #lag features for time-series dependency
    df['lag_1'] = df['Average_Temperature_C'].shift(1)
    df['lag_12'] = df['Average_Temperature_C'].shift(12)
    df.dropna(inplace=True)             # delete all null values from lags
    
    fig, ax = plt.subplots(figsize=(10,8))
    
    numeric_variables = df[['Average_Temperature_C', 
                            'Total_Rainfall_mm',
                            'Max_Temperature_C',
                            'Min_Temperature_C',
                            'month_sin',    #cyclical month(sine)
                            'month_cos',    #cyclical month (cosine)
                            'lag_1',
                            'lag_12']] 
    
    seascorr = numeric_variables.corr()

    sns.heatmap(
            seascorr, 
            cmap='coolwarm', 
            annot=True, 
            vmin=-1, 
            vmax=1,
            fmt=".2f",
            linewidths=0.5,
            mask = np.triu(np.ones_like(seascorr, dtype=bool), k=1),  #lower triangle only
            ax=ax)
    ax.set_title("Correlation Heatmap: Temperature and Rainfall (Monthly Patterns)")
    plt.tight_layout()
    return fig 

def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plot the actual vs predicted values
    """
    fig, ax = plt.subplots(figsize = (10,6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    ax.set_xlabel("Actual Avg Temperature")
    ax.set_ylabel("Predicted Avg Temperature")
    ax.set_title("Actual vs Predicted Avg Temperatures")
    ax.grid(True)
    
    return fig

def plot_forecast(forecast):
    """
    Plot the predicted Avg Temp, forecast Min Temp and Max Temp 
    """
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast['Month'], forecast['Predicted_Avg_Temp'], label='Avg Temp', marker='o')
    ax.plot(forecast['Month'], forecast['Forecasted_Min_Temp'], label='Min Temp', linestyle='--')
    ax.plot(forecast['Month'], forecast['Forecasted_Max_Temp'], label='Max Temp', linestyle='--')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend()
    
    return fig

'''
def plot_prediction_context():

    """
    Plot the actual vs predicted values
    """
    fig, ax = plt.subplots(figsize = (10,6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    ax.set_xlabel("Actual Avg Temperature")
    ax.set_ylabel("Predicted Avg Temperature")
    ax.set_title("Actual vs Predicted Avg Temperatures")
    ax.grid(True)
    
    return fig
'''