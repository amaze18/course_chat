import streamlit as st
import os
import boto3
from botocore.exceptions import NoCredentialsError
s3_bucket_name="coursechat"
access_key = os.environ.get('ACCESS_ID')
secret_key = os.environ.get('ACCESS_KEY')
# Set your AWS credentials and region (replace with your own values)
AWS_ACCESS_KEY = access_key
AWS_SECRET_KEY =secret_key
S3_BUCKET_NAME = s3_bucket_name

def create_s3_subfolder(course_name):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    # Create a new subfolder under the specified bucket
    subfolder_path = f"{course_name}/"
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=subfolder_path)

def upload_to_s3(course_name, file):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    subfolder_path = f"{course_name}/"
    s3_file_path = f"{subfolder_path}{file.name}"
    try:
        s3.upload_fileobj(file, S3_BUCKET_NAME, s3_file_path)
        st.success(f"File '{file.name}' uploaded successfully to '{subfolder_path}'")
    except NoCredentialsError:
        st.error("AWS credentials not available. Please check your credentials.")

def main():
    st.title("Course Management App")

    # Sidebar
    action = st.sidebar.selectbox("Select Action", ["Create New Course", "Upload Files"])

    if action == "Create New Course":
        create_new_course()
    elif action == "Upload Files":
        upload_files()

def create_new_course():
    st.header("Create a New Course")

    # Type box to get input for course name
    course_name = st.text_input("Enter course name:")

    if st.button("Create Course"):
        # Create a new subfolder in S3
        create_s3_subfolder(course_name)

        # Store the course name as a variable
        st.success(f"Course '{course_name}' created successfully! Subfolder in S3 created.")

def upload_files():
    st.header("Upload Files")

    # Type box to get input for course name
    course_name = st.text_input("Enter course name:")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv"])

    if st.button("Upload File") and uploaded_file:
        # Upload file to S3
        upload_to_s3(course_name, uploaded_file)
    import Index
if __name__ == "__main__":
    main()
