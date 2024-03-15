import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
import pandas as pd
from io import StringIO
import boto3
import os

st.cache_data.clear()

#allowed_emails_csv_path = "allowed_emails.csv"
allowed_emails_df = pd.read_csv('allowed_emails.csv')
allowed_emails_set = set(allowed_emails_df['email'].str.lower())

__login__obj = __login__(auth_token = "Courier_auth_api", 
                    company_name = "Evaluatingcodes",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json',
                    allowed_emails_set= allowed_emails_set)
                    
LOGGED_IN = __login__obj.build_login_ui()

if LOGGED_IN == True:

    st.write("Your Streamlit Application Begins here!")
    text_input = st.text_input(
        "Enter course name 👇",
    #    label_visibility=st.session_state.visibility,
    #    disabled=st.session_state.disabled,
     #   placeholder=st.session_state.placeholder,
    )

    if text_input:
        st.write("You entered: ", text_input)

    S3_BUCKET_NAME = "your_bucket_name"
    AWS_ACCESS_KEY_ID = "your_access_id"
    AWS_SECRET_ACCESS_KEY = "your_access_key"    

    #uploaded_file = st.file_uploader("Choose a file")
    uploaded_filenames=[]
    def upload():
        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        #uploaded_filenames=[]
        for uploaded_file in uploaded_files:
           bytes_data = uploaded_file.read()
           #st.write("filename:", uploaded_file.name)
           #st.write(bytes_data)
           local_file_path = os.path.join("temp", uploaded_file.name)
           with open(local_file_path, "wb") as local_file:
                local_file.write(bytes_data)
           s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            )

           #for uploaded_files  in uploaded_files:
           s3.upload_file(local_file_path ,S3_BUCKET_NAME, uploaded_file.name)
           uploaded_filenames.append(uploaded_file.name)
           #st.write(bytes_data)
           os.remove(local_file_path)

        #st.write("Uploaded files: " ,uploaded_filenames)   
        st.write("Uploaded filenames:", ', '.join(uploaded_filenames))   
    upload()    

