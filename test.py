import streamlit as st
import pandas as pd

# Function to check if the user's email exists in the CSV file
def check_user(email):
    users_df = pd.read_csv("teacher.csv")  # Assuming users.csv contains a column named "email"
    if email in users_df["email"].tolist():
        return True
    else:
        return False
    
def load_courses(file_path):
    courses = pd.read_csv(file_path)["course"].tolist()
    return courses

# MAIN FUNCTION
def main():
    # Check if user is logged in
    logged_in = False  # Set to True if user is logged in
    user_email = "Anupam.ainml@gmail.com"  # Placeholder for user's email, replace this with actual user email

    if check_user(user_email):
        logged_in = True

    if logged_in:
        action = st.selectbox("Select Action", ["Create New course", "Update an existing course"])
        
        if action == "Create New course":
            course_name = st.text_input("Course name:")
            # Function to handle creation of new course
            # upload_files(course_name)
        
        elif action == "Update an existing course":
            course_name = st.text_input("Course name:")
            # Function to handle updating existing course
            # upload_files(course_name)
        
        
    else:
        action = st.selectbox("Select Action", ["Course chat"])
        if action == "Course chat":
            courses = load_courses("teacher.csv")
            action = st.selectbox("Select Course:", courses)
            
            # Function to handle course chat
            # course_chat(option)

if __name__ == "__main__":
    main()
