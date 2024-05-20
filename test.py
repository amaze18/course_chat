import psycopg2
import pandas as pd

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname="app_login_db",
    user="cuser",
    password="123",
    host="localhost",
    port="5432"
)

# Read CSV data into DataFrame
df = pd.read_csv('your_csv_file.csv')

# Create a cursor object
cur = conn.cursor()

# Insert data into PostgreSQL table
for index, row in df.iterrows():
    cur.execute(
        "INSERT INTO your_table_name (email, restricted) VALUES (%s, %s)",
        (row['email'], row['restricted'])
    )

# Commit and close
conn.commit()
cur.close()
conn.close()
