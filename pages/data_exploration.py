import streamlit as st
from visualizations import plot_time_series, heatmap_corr, seasonal_temperature, seasonal_rainfall, seasonal_decompose

def show(df):
    """
    Display the data exploration page
    """
    st.header("Data Exploration")
    st.dataframe(df.head())

    #Show the raw data
    st.subheader("Statistical Summary")
    st.write(df[['Average_Temperature_C', 'Total_Rainfall_mm', 'Max_Temperature_C', 'Min_Temperature_C']].describe())

    #Time Series Plot
    st.subheader("Temperature over time")
    fig = plot_time_series(df)
    st.pyplot(fig)

    #Plot the seasonal Temperature 
    st.subheader("Seasonal Temperature Patterns")
     
    fig1, fig2 = seasonal_temperature(df)
    # display the figures in separate containers
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)
   

    #Plot the seasonal ranfall 
    st.subheader("Seasonal Rainfall Patterns")
    
    fig1, fig2 = seasonal_rainfall(df)
    
    # display the figures in separate containers
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)
   
   #Plot the heatmap
    st.subheader("Heatmap for Temperature and Rainfall over the years")
    fig = heatmap_corr(df)
    st.pyplot(fig)

    