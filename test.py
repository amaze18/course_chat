import pandas as pd

def check_instructor_mode(csv_file_path, email):
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Check if the email is present in the DataFrame
        if email in df["email"].tolist():
            # Get the mode corresponding to the email
            mode = df.loc[df["email"] == email, "mode"].iloc[0]
            # Check if mode is "instructor"
            if mode == "instructor":
                return True
            else:
                return False
        else:
            print(f"Email '{email}' not found in the CSV file.")
            return False
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

# Example usage:
teachers_csv_path = "allowed_emails.csv"
email_to_check = "saikbulusu@gmail.com"
result = check_instructor_mode(teachers_csv_path, email_to_check)
print(result)  # True or False
