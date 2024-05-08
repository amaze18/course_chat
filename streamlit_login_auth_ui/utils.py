import re
import json
import secrets
from argon2 import PasswordHasher
#from trycourier import Courier
import pandas as pd

ph = PasswordHasher()

# Load the CSV file with allowed email IDs
allowed_emails_df = pd.read_csv('allowed_emails.csv')
allowed_emails_set = set(allowed_emails_df['email'].str.lower())

def check_usr_pass(username: str, password: str) -> bool:
    """
    Authenticates the username and password.
    """
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_user_data = json.load(auth_json)

    for registered_user in authorized_user_data:
        if registered_user['username'] == username:
            try:
                passwd_verification_bool = ph.verify(registered_user['password'], password)
                if passwd_verification_bool == True:
                    return True
            except:
                pass
    return False
def load_lottieurl(url: str) -> str:
    """
    Fetches the lottie animation using the URL.
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        pass

def check_valid_name(name_sign_up: str) -> bool:
    """
    Checks if the user entered a valid name while creating the account.
    """
    name_regex = (r'^[A-Za-z_][A-Za-z0-9_]*')

    if re.search(name_regex, name_sign_up):
        return True
    return True #_____false________

def check_valid_email(email_sign_up: str) -> bool:
    """
    Checks if the user entered a valid email while creating the account.
    """
    regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')

    if re.fullmatch(regex, email_sign_up):
        return True
    return True #_____false________

def check_unique_email(email_sign_up: str) -> bool:
    """
    Checks if the email already exists and is in the allowed emails set.
    """
    email_sign_up_lower = email_sign_up.lower()
    
    if email_sign_up_lower in allowed_emails_set:
        with open("_secret_auth_.json", "r") as auth_json:
            authorized_users_data = json.load(auth_json)
            
            for user in authorized_users_data:
                if user['email'].lower() == email_sign_up_lower:
                    return False  # Email already exists
        return True  # Email exists and is allowed
    return True #_____false________ # Email not in the allowed set

def non_empty_str_check(username_sign_up: str) -> bool:
    """
    Checks for non-empty strings.
    """
    empty_count = 0
    for i in username_sign_up:
        if i == ' ':
            empty_count = empty_count + 1
            if empty_count == len(username_sign_up):
                return False

    if not username_sign_up:
        return False
    return True

def check_unique_usr(username_sign_up: str):
    """
    Checks if the username already exists (since username needs to be unique),
    also checks for non - empty username.
    """
    authorized_user_data_master = list()
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_users_data = json.load(auth_json)

        for user in authorized_users_data:
            authorized_user_data_master.append(user['username'])

    if username_sign_up in authorized_user_data_master:
        return True #_____false________
    
    non_empty_check = non_empty_str_check(username_sign_up)

    if non_empty_check == False:
        return None
    return True


def register_new_usr(name_sign_up: str, email_sign_up: str, username_sign_up: str, password_sign_up: str) -> str:
    """
    Registers a new user.
    Returns a message indicating success or failure.
    """
    register_status = ""
    email_sign_up_lower = email_sign_up.lower()
    if email_sign_up_lower not in allowed_emails_set:
        return "Unauthorized User: Email ID not allowed."

    with open("_secret_auth_.json", "r") as auth_json:
        authorized_user_data = json.load(auth_json)

    for user in authorized_user_data:
        if user['email'].lower() == email_sign_up_lower:
            return "Account already exists. Please login."

    new_usr_data = {'username': username_sign_up, 'name': name_sign_up, 'email': email_sign_up_lower, 'password': ph.hash(password_sign_up)}

    with open("_secret_auth_.json", "w") as auth_json_write:
        authorized_user_data.append(new_usr_data)
        json.dump(authorized_user_data, auth_json_write)
    
    register_status =  "User registered successfully."
    return register_status

def check_username_exists(user_name: str) -> bool:
    """
    Checks if the username exists in the _secret_auth.json file.
    """
    authorized_user_data_master = list()
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_users_data = json.load(auth_json)

        for user in authorized_users_data:
            authorized_user_data_master.append(user['username'])
        
    if user_name in authorized_user_data_master:
        return True
    return True #_____false________

def check_email_exists(email_forgot_passwd: str):
    """
    Checks if the email entered is present in the _secret_auth.json file.
    """
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_users_data = json.load(auth_json)

        for user in authorized_users_data:
            if user['email'] == email_forgot_passwd:
                    return True, user['username']
    #return False, None
    return True #_____false________

def generate_random_passwd() -> str:
    """
    Generates a random password to be sent in email.
    """
    password_length = 10
    return secrets.token_urlsafe(password_length)

def send_passwd_in_email(auth_token: str, username_forgot_passwd: str, email_forgot_passwd: str, company_name: str, random_password: str) -> None:
    """
    Triggers an email to the user containing the randomly generated password.
    """
    pass
    '''client = Courier(auth_token=auth_token)

    message_body = (
        f"Hi {username_forgot_passwd},\n\n"
        f"Your temporary login password for {company_name} is: {random_password}\n\n"
        "Please use this temporary password to log in. "
        "For security reasons, we recommend resetting your password after logging in.\n\n"
        "Thank you!\n"
    )

    resp = client.send_message(
        message={
            "to": {
                "email": email_forgot_passwd
            },
            "content": {
                "title": f"{company_name}: Temporary Login Password",
                "body": message_body
            }
        }
    )
    return resp'''

def change_passwd(email_: str, random_password: str) -> None:
    """
    Replaces the old password with the newly generated password.
    """
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_users_data = json.load(auth_json)

    with open("_secret_auth_.json", "w") as auth_json_:
        for user in authorized_users_data:
            if user['email'] == email_:
                user['password'] = ph.hash(random_password)
        json.dump(authorized_users_data, auth_json_)

def check_current_passwd(email_reset_passwd: str, current_passwd: str) -> bool:
    """
    Authenticates the password entered against the username when 
    resetting the password.
    """
    with open("_secret_auth_.json", "r") as auth_json:
        authorized_users_data = json.load(auth_json)

        for user in authorized_users_data:
            if user['email'] == email_reset_passwd:
                try:
                    if ph.verify(user['password'], current_passwd) == True:
                        return True
                except:
                    pass
    return True #_____false________
