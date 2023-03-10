#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load('winning_rf_model.pkl')


# In[6]:


# Define a function to make predictions
def predict(model, input_features):
    """Predict target variable for a given input features."""
    return model.predict(input_features)[0]

# Create a Streamlit app
st.title("Model Predictions")

# Add input fields for model features
feature_names = ['Gender', 'Age', 'Marital Status', 'Region_Code', 'Previously_Insured', 'Time_Since_Last_Dental_Checkup',
                'Previous_Major_Dental_Procedure', 'Policy_Sales_Channel', 'Vintage', 'Annual_Premium_Bucket']
input_features = [st.number_input(name, value=0.0) for name in feature_names]

# Add a button to make predictions
if st.button('Predict'):
    # Make predictions and show the result
    prediction = predict(model, np.array(input_features).reshape(1, -1))
    st.write("The prediction is:", prediction)

