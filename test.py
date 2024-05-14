import csv

def check_blocked_email(email, csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['email'] == email:
                if row['access']:
                    return True, row['access']
                else:
                    return False, "Access not blocked"
        return False, "Email not found"

# Example usage
email_to_check = "sudipta2020@gmail.com"
csv_file_path = "allowed_emails.csv"

blocked, reason = check_blocked_email(email_to_check, csv_file_path)
if blocked:
    print(f"The email {email_to_check} is blocked. Reason: {reason}")
else:
    print(f"The email {email_to_check} is not blocked.")
