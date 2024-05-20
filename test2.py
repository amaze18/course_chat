import pandas as pd

def get_course_list_from_csv(file_path, email):
    try:
        df = pd.read_csv(file_path)
        # Filter rows where the email matches
        email_courses = df[df['email'] == email]["course"].tolist()
        return email_courses
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

# Example usage:
courses = get_course_list_from_csv('teachers.csv', 'Anupam.aiml@gmail.com')
print(courses)
