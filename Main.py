import streamlit as st
import pandas as pd
import numpy as np

import pickle
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load model
model = pickle.load(open('model_xgb.pkl', 'rb'))


# Title
st.title('Loan Default Prediction')

st.write('Fill out the form: ')

col1, col2 = st.columns(2)

with col1:
   person_age = st.number_input("Input Age", value=0)

with col1:
   person_emp_exp = st.number_input("Input employment experience (years)", value=0)

with col1:
   cb_person_cred_hist_length = st.number_input("Input length of the applicant's credit history (years)", value=0)

with col1:
   person_income = st.number_input("Input annual income of the applicant (USD)", value=0)

with col2:
   loan_amnt = st.number_input("Input loan amount requested (in USD)", value=0)

with col2:
   loan_percent_income = st.number_input("Input ratio of loan amount to income",value=0.0, step=0.1)

with col2:
   loan_int_rate = st.number_input("Input interest rate on the loan (percentage)",value=0.0, step=0.1)

with col2:
   credit_score = st.number_input("Input credit score of the applicant.", value=0)


# Pastikan semua variabel yang digunakan dalam prediksi berupa angka (int/float)
if st.button('Prediction'):
    pred_peminjaman = model.predict([[
        person_age,
        person_emp_exp,
        cb_person_cred_hist_length,
        loan_amnt,
        loan_percent_income,
        person_income,
        loan_int_rate,
        credit_score

    ]])

    st.write(f'Result: {pred_peminjaman[0]}') 

    if pred_peminjaman == 0 :
        st.error('The applicant defaulted')
    elif pred_peminjaman == 1 :
        st.success('Loan Was Repaid Successfully') 
    else:
        st.error('Unpredictable')