import pickle
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np


class columnDropperTransformer():
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self

pickle_in = open('preprocessor.pkl', 'rb')
preprocessor = pickle.load(pickle_in)
model = tf.keras.models.load_model('model.h5')




def predict(inputs):
    # preprocessor

    # input --> model training
    # SAMA PERSIS
    df = pd.DataFrame(inputs, index=[0])
    df = preprocessor.transform(df)

    y_pred = model.predict(df)
    y_pred = np.where(y_pred < 0.5, 0, 1).squeeze()
    return y_pred.item()



# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Churn Prediction</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    template = {
        'customer_id': [],
        'gender': [],
        'senior_citizen': [],
        'partner': [],
        "dependents": [],
        "tenure": [],
        "phone_service": [],
        "multiple_lines": [],
        "internet_service": [],
        "online_security": [],
        "online_backup": [],
        "device_protection": [],
        "tech_support": [],
        "streaming_tv": [],
        "streaming_movies": [],
        "contract": [],
        "paperless_billing": [],
        "payment_method": [],
        "monthly_charges": [],
        "total_charges": []
    }

    log = pd.DataFrame({
        'result': [],
        'customer_id': [],
        'gender': [],
        'senior_citizen': [],
        'partner': [],
        "dependents": [],
        "tenure": [],
        "phone_service": [],
        "multiple_lines": [],
        "internet_service": [],
        "online_security": [],
        "online_backup": [],
        "device_protection": [],
        "tech_support": [],
        "streaming_tv": [],
        "streaming_movies": [],
        "contract": [],
        "paperless_billing": [],
        "payment_method": [],
        "monthly_charges": [],
        "total_charges": []
    }
    )
    if 'log' not in st.session_state:
        st.session_state.log = log

    col1, col2, col3 = st.columns(3)
    inputs = template.copy()

    col1.header('Customer Information')
    inputs['customer_id'] = col1.text_input('Your ID', 'xxxx-xxxx')
    inputs['gender'] = col1.radio("Your Gender", ('Male', 'Female'))
    senior_citizen = col1.radio("Are you a senior?", ('Yes', 'No'))

    inputs['senior_citizen'] = 1 if senior_citizen == "Yes" else 0

    inputs['partner'] = col1.radio("Do you have a partner?", ('Yes', 'No'))
    inputs['dependents'] = col1.radio(
        "Do you have a dependent?", ('Yes', 'No'))

    col2.header('Account Information')
    inputs['tenure'] = col2.number_input(
        'How long have you been with us? (month)', min_value=0, max_value=200)
    inputs['contract'] = col2.selectbox(
        'Contract type', ("Month-to-month", "One year", "Two year"))
    inputs['paperless_billing'] = col2.radio(
        "Using Paperless Billing?", ('Yes', 'No'))
    inputs['payment_method'] = col2.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                                                 'Credit card (automatic)'))
    inputs['monthly_charges'] = col2.number_input(
        'How much do you pay a month?', min_value=0, max_value=200)
    inputs['total_charges'] = col2.number_input(
        'How much do you pay in total?', min_value=0, max_value=20000)

    col3.header('Services Information')
    inputs['phone_service'] = col3.radio(
        "Do you have phone service?", ('Yes', 'No'))

    inputs['multiple_lines'] = col3.selectbox(
        'Do you have multiple lines?', ('No phone service', 'No', 'Yes'))
    inputs['internet_service'] = col3.selectbox(
        'Do you have internet service?', ('DSL', 'Fiber optic', 'No'))

    inputs['online_security'] = col3.selectbox('Do you subscribe to our online security service?',
                                               ('No', 'Yes', 'No internet service'))

    inputs['online_backup'] = col3.selectbox('Do you subscribe to our online backup service?',
                                             ('No', 'Yes', 'No internet service'))
    inputs['device_protection'] = col3.selectbox('Do you subscribe to our device protection service?',
                                                 ('No', 'Yes', 'No internet service'))
    inputs['tech_support'] = col3.selectbox('Do you subscribe to our tech support service?',
                                            ('No', 'Yes', 'No internet service'))
    inputs['streaming_tv'] = col3.selectbox('Do you subscribe to our streaming tv service?',
                                            ('No', 'Yes', 'No internet service'))
    inputs['streaming_movies'] = col3.selectbox('Do you subscribe to our streaming movies service?',
                                                ('No', 'Yes', 'No internet service'))

    # rewrite to
    if st.button("Predict"):
        # pass
        inputs['result'] = predict(inputs)

        if inputs['result'] == 0:
            st.success(f"The Customer will not churn!")
        else:
            st.error(f"The Customer will churn!")

        st.session_state.log = st.session_state.log.append(
            inputs, ignore_index=True)
    st.dataframe(st.session_state.log)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    main()
