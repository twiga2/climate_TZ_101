import streamlit as st
import numpy as np
from model_utils import load_model
from visualizations import plot_forecast
from prediction import make_prediction, get_historical_average

def show(df):
    """Display the prediction page"""
    st.header("Temperature Predictions")

    #Check if the model exists
    if 'model' not in st.session_state:
        loaded_model = load_model()
        if loaded_model is None:
            st.warning("No trained Model found. Please go to the model training page first")
            st.stop()
        st.session_state['model'] = loaded_model
        st.session_state['model_type'] = "Pre-trained model"
    #access the model from session_state
    model = st.session_state['model']

    #Show the model that is being used
    st.info(f"Using {st.session_state['model_type']} for predictions")

    # Prediction input
    st.subheader("Select the date for prediction")
    pred_year = st.slider("Year", 2021, 2030, 2025)
    #pred_month = st.slider("Month", 1, 12, 6)

    # Make prediction
    if st.button("Predict Temperature"):
        #Get model
        historical_avg = get_historical_average(df)  # gives forecast Max and Min Temp
        #model = st.session_state['model']
        # Make prediction
        forecast = make_prediction(model, pred_year, historical_avg)

        #Display the results
        #st.sucess(f"Predicted temperature for {pred_year}-{pred_month:02d} : {prediction:.2f} C")
        st.write(forecast)  
       
        # Visualize
        st.subheader("Predicted Avg Temperarute and Forecast Max and Min Temperature")
        # plot prediction context
        fig = plot_forecast(forecast)
        st.pyplot(fig)

    
   
    '''
    # Give the historical comparison
    hist_avg = get_historical_average(df, pred_month)
    
    st.write(f"Historical average for month {pred_month}: {hist_avg:.2f}C")   

    # Calculate the difference
    diff = prediction - hist_avg
    if diff > 0:
        st.write(f"Prediction is {diff:.2f}C **higher** than historical average") 
    else:
        st.write(f"Prediction is {abs(diff):.2f}C **lower** than historical average")
   

    # Visualize
    st.subheader("Prediction in the Historical Context")

    #get historical context
    hist_temps = get_historical_context(df, pred_month)

    # plot prediction context
    fig = plot_prediction_context(hist_temps, pred_year, pred_month, prediction)
    st.pyplot(fig)
    '''