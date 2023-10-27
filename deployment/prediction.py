import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

def runPredictor():
    #MODEL
    with open('model_knn.pkl', 'rb') as file_1:
        model_knn = pickle.load(file_1)

    #SCALER
    with open('model_scaler_mm.pkl', 'rb') as file_2:
        model_scaler_mm = pickle.load(file_2)

    #ENCODER
    with open('model_oh_encoder.pkl', 'rb') as file_3:
        model_oh_encoder = pickle.load(file_3)

    #NUMERICAL
    with open('list_num_column_skew.txt', 'rb') as file_4:
        list_num_column_skew = pickle.load(file_4)

    #CATEGORICAL
    with open('list_cat_column_ordinal.txt', 'rb') as file_5:
        list_cat_column_ordinal = pickle.load(file_5)

    with open('list_cat_column_nominal.txt', 'rb') as file_6:
        list_cat_column_nominal = pickle.load(file_6)

    # Buat Form
    with st.form(key='Form Parameters'):
        limit_balance = st.number_input('Limit Balance', min_value=0, max_value=1000000, step=5000)
        # age = st.number_input('Age', min_value=0, max_value=99, step=1, help='Usia')
        age_cat =  st.selectbox('Age Category', ('Children', 'Young Adult', 'Adult', 'Middle Age', 'Old Age'), index=0, help='1-16 : Children, 17-30 : Young Adult, 31-40 : Adult, 41-50 : Middle Age, 50+ Old Age')
        sex =  st.selectbox('Sex', ('Male', 'Female'), index=0)
        education_level = st.selectbox('Education Level', ('Graduate School', 'University', 'High School', 'Others', 'Unknown'), index=0)
        marital_status =  st.selectbox('Marital Status', ('Married', 'Single', 'Others'), index=0)
        st.markdown('---')
        pay_1 = st.selectbox('Repayment in September', ('-2', '-1', '0', '1', '2', '3', '4', '5', '6'), index=2)
        pay_2 = st.selectbox('Repayment in August', ('-2', '-1', '0', '1', '2', '3', '4', '5'), index=2)
        pay_3 = st.selectbox('Repayment in July', ('-2', '-1', '0', '1', '2', '3', '4'), index=2)
        pay_4 = st.selectbox('Repayment in June', ('-2', '-1', '0', '1', '2', '3'), index=2)
        pay_5 = st.selectbox('Repayment in May', ('-2', '0', '2', '3'), index=1)
        pay_6 = st.selectbox('Repayment in April', ('-2', '0','1', '2', '3', '4'), index=1)
        st.markdown('---')
        bill_amt_1 = st.number_input('Bill Amount1', min_value=-1000000, max_value=1000000, step=50000)
        bill_amt_2 = st.number_input('Bill Amount2', min_value=-1000000, max_value=1000000, step=50000)
        bill_amt_3 = st.number_input('Bill Amount3', min_value=-1000000, max_value=1000000, step=50000)
        bill_amt_4 = st.number_input('Bill Amount4', min_value=-1000000, max_value=1000000, step=50000)
        bill_amt_5 = st.number_input('Bill Amount5', min_value=-1000000, max_value=1000000, step=50000)
        bill_amt_6 = st.number_input('Bill Amount6', min_value=-1000000, max_value=1000000, step=50000)
        st.markdown('---')
        pay_amt_1 = st.number_input('Pay Amount1', min_value=-1000000, max_value=1000000, step=50000)
        pay_amt_2 = st.number_input('Pay Amount2', min_value=-1000000, max_value=1000000, step=50000)
        pay_amt_3 = st.number_input('Pay Amount3', min_value=-1000000, max_value=1000000, step=50000)
        pay_amt_4 = st.number_input('Pay Amount4', min_value=-1000000, max_value=1000000, step=50000)
        pay_amt_5 = st.number_input('Pay Amount5', min_value=-1000000, max_value=1000000, step=50000)
        pay_amt_6 = st.number_input('Pay Amount6', min_value=-1000000, max_value=1000000, step=50000)

        submitted = st.form_submit_button('Predict')

        data_inf = {'limit_balance': limit_balance,
        'age_cat': age_cat,
        'sex': sex,
        'education_level': education_level,
        'marital_status': marital_status,
        'pay_1': int(pay_1),
        'pay_2': int(pay_2),
        'pay_3': int(pay_3),
        'pay_4': int(pay_4),
        'pay_5': int(pay_5),
        'pay_6': int(pay_6),
        'bill_amt_1': bill_amt_1,
        'bill_amt_2': bill_amt_2,
        'bill_amt_3': bill_amt_3,
        'bill_amt_4': bill_amt_4,
        'bill_amt_5': bill_amt_5,
        'bill_amt_6': bill_amt_6,
        'pay_amt_1': pay_amt_1,
        'pay_amt_2': pay_amt_2,
        'pay_amt_3': pay_amt_3,
        'pay_amt_4': pay_amt_4,
        'pay_amt_5': pay_amt_5,
        'pay_amt_6': pay_amt_6}

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # Split between num column and cat column
        data_inf_num = data_inf[list_num_column_skew]
        data_inf_cat_o = data_inf[list_cat_column_ordinal]
        data_inf_cat_n = data_inf[list_cat_column_nominal]

        #scaling data
        data_inf_num_scaled = model_scaler_mm.transform(data_inf_num)
        #encoding data
        data_inf_cat_n_encoded = model_oh_encoder.transform(data_inf_cat_n).toarray()

        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_o, data_inf_cat_n_encoded], axis=1)

        threshold = 0.6
        y_pred_new_proba_train = model_knn.predict_proba(data_inf_final)
        y_pred_new_train = np.where(y_pred_new_proba_train[:,1] >= threshold, 1, 0)

        st.write(f'# Predict Result : {str(int(y_pred_new_train))}')

if __name__ == '__main__':
    runPredictor()