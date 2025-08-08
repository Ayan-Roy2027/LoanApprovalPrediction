import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st

#load the model
model2 = load_model('model2.h5')

#Save encoders and Ohe
with open('le_edu.pkl','rb') as file:
    le_edu = pickle.load(file)
with open('le_employed.pkl','rb') as file:
    le_employed = pickle.load(file)
with open('le_gender.pkl','rb') as file:
    le_gender = pickle.load(file)
with open('le_married.pkl','rb') as file:
    le_married = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
with open('ohe_prop.pkl','rb') as file:
    ohe_prop = pickle.load(file)

#Streamlit app
st.title('Loan approving prediction')

#User Input
gender = st.selectbox('Gender',le_gender.classes_)
married = st.selectbox('Married',le_married.classes_)
dependents = st.slider('Dependents',0,20)
education = st.selectbox('Education',le_edu.classes_)
employed_status = st.selectbox('SelfEmployed',le_employed.classes_)
income = st.number_input('Income of applicant')
income_ = st.number_input('Income of co-applicant')
loan_amount = st.number_input('Loan Amount')
loan_months = st.number_input('LoanTermMonths',step=1)
credit = st.slider('CreditHistory',0,1)
property_ = st.selectbox('PropertyArea',ohe_prop.categories_[0])

#Input data
input_data = pd.DataFrame({
    'Gender': [le_gender.transform([gender])[0]],               
    'Married': [le_married.transform([married])[0]],               
    'Dependents': [dependents],              
    'Education': [le_edu.transform([education])[0]],        
    'SelfEmployed': [le_employed.transform([employed_status])[0]],          
    'ApplicantIncome': [income],        
    'CoapplicantIncome': [income_],      
    'LoanAmount': [loan_amount],              
    'LoanTermMonths': [loan_months],          
    'CreditHistory': [credit],             
})

#OHE propertyArea
encoded_prop = ohe_prop.transform([[property_]]).toarray()
column = ohe_prop.get_feature_names_out(['PropertyArea'])
prop_df = pd.DataFrame(encoded_prop,columns=column)

input_data = pd.concat([input_data.reset_index(drop=True),prop_df],axis=1)

#Scaling the dataframe
input_scaled = scaler.transform(input_data)

#Click button
if st.button("Predict"):
        #predict
    prediction = model2.predict(input_scaled)
    prediction_proba = prediction[0][0]
    if prediction_proba > 0.5:
        st.write('Eligible for loan')
    else:
        st.write('Not eligible for loan')

