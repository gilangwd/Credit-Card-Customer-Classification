import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Select Page :', ('EDA', 'Predict Credit Card Default'))

if navigation == 'EDA':
    eda.runEDA()
else:
    prediction.runPredictor()

#streamlit run app.py