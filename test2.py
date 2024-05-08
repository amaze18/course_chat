import mysql.connector

# Connect to MySQL database
try:
    db_connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="instructor"
    )

    # Create a cursor object
    cursor = db_connection.cursor()

    # Define SQL statement to create table
    create_table_query = """
    CREATE TABLE instructors (
        email VARCHAR(255),
        course VARCHAR(255)
    )
    """

    # Execute SQL statement to create table
    cursor.execute(create_table_query)
    print("Table 'instructors' created successfully.")

except mysql.connector.Error as error:
    print("Error creating table:", error)

finally:
    # Close cursor and database connection
    if 'cursor' in locals():
        cursor.close()
    if 'db_connection' in locals():
        db_connection.close()
