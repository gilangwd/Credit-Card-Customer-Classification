import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title='Credit Card Default - EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)

def runEDA():
    #Title
    st.title('Credit Card Default Prediction')

    #Sub Header
    st.subheader('EDA for Credit Card Default Prediction')

    #Description
    st.write('Page Created by Gilang Wiradhyaksa (SBY-001)')

    st.markdown('---')

    '''
    On this Page we will do a simple exploration,
    Used Database is Credit Card Default Payment.
    Dataset Source is from Google Bigquery
    '''

    #show dataframe
    st.title('Dataset')
    df = pd.read_csv('P1G5_Set_1_gilang.csv')
    st.dataframe(df)

    st.write('## Histogram Limit Balance')
    fig = plt.figure(figsize=(15,5))
    sns.histplot(df['limit_balance'], bins=20, kde=True).set(title='limit_balance')
    st.pyplot(fig)
    st.write('Based on histogram, column _limit\_balance_ skewness is positive, meaning the data distribution is not normal.')
    st.markdown('---')

    st.write('## Average Amount of Bill Statement')
    df_amt = pd.DataFrame()
    bill_amt = []
    pay_amt = []
    for i in range(1, 7):
        bill_amt.append(df['bill_amt_' + str(i)].mean())
        pay_amt.append(df['pay_amt_' + str(i)].mean())
    df_amt['bill_amt'] = bill_amt
    df_amt['pay_amt'] = pay_amt
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    axis_label = sns.barplot(ax=ax[0], data=df_amt, x=df_amt['bill_amt'].index, y=bill_amt, orient='v')
    ax[0].set_title('Bill Statement')
    axis_label = sns.barplot(ax=ax[1], data=df_amt, x=df_amt['pay_amt'].index, y=pay_amt, orient='v')
    ax[1].set_title('Payment')
    st.pyplot(fig)
    st.write('Average amount of bill statement is decrease every month, showing people using their credit card less during this period.')
    st.markdown('---')

    st.write('## Barplot Sex')
    fig = plt.figure(figsize=(10,5))
    sns.countplot(x='sex', data=df)
    st.pyplot(fig)
    st.write('Most of the bank customer is female')
    st.markdown('---')

    st.write('## Barplot Marital Status')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(x='marital_status', data=df)
    st.pyplot(fig)
    st.write('Most of the bank customer is married')
    st.markdown('---')

if __name__ == '__main__':
    runEDA()