#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Load the model
model = joblib.load('winning_rf_model.pkl')


# In[6]:


# Define a function to make predictions
def predict(model, input_features):
    """Predict target variable for a given input features."""
    return model.predict(input_features)[0]

# Create a Streamlit app
st.title("Dental Prospect Predictor")
st.header('Enter the characteristics of the member:')
st.image("""https://www.bing.com/images/search?view=detailV2&ccid=AWme%2bOGw&id=87D6CF795F18AE7DC9305FB72E7A0EA186F744A4&thid=OIP.AWme-OGwlRjaY9OGsLbrtAHaCJ&mediaurl=https%3a%2f%2fcontent-static.healthcare.inc%2fuploads%2fsites%2f2%2f2021%2f03%2fAetna-Logo-PNG-Transparent-2.png&cdnurl=https%3a%2f%2fth.bing.com%2fth%2fid%2fR.01699ef8e1b09518da63d386b0b6ebb4%3frik%3dpET3hqEOei63Xw%26pid%3dImgRaw%26r%3d0&exph=900&expw=3100&q=aetna&simid=608037601425114515&FORM=IRPRST&ck=A7EF76EB3BB273FAEC157DFD3117DD04&selectedIndex=0""")


# Add input fields for model features
feature_names = ['Gender', 'Age', 'Marital Status', 'Region_Code', 'Previously_Insured', 'Time_Since_Last_Dental_Checkup',
                'Previous_Major_Dental_Procedure', 'Policy_Sales_Channel', 'Vintage', 'Annual_Premium_Bucket']
input_features = [st.number_input(name, value=0.0) for name in feature_names]

# Add a button to make predictions
if st.button('Predict'):
    # Make predictions and show the result
    prediction = predict(model, np.array(input_features).reshape(1, -1))
    st.write("The prediction is:", prediction)

