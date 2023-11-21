# OS traversal 
import os 
    
# SQLite connection 
import sqlite3

# Defining the processed_images table 
# The columns are
# image_name (str)
# datetime_processed (datetime)
# prediction_prob (float)
# prediction_class (str)

if __name__ == '__main__':
    # Infering the current file path 
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Connecting 
    conn = sqlite3.connect(os.path.join(current_dir, 'database.db'))

    # Creating the table
    cursor = conn.cursor()

    # Checking if the table exists
    cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="processed_images"')

    # Fetching the result
    result = cursor.fetchall()

    # Checking if the table exists
    if len(result) > 0:
        # Exiting
        print('The table processed_images already exists')
        exit(0)

    # Creating the table
    cursor.execute("""
        CREATE TABLE processed_images (
            image_name text,
            datetime_processed datetime,
            prediction_prob float,
            prediction_class text
        )
    """)

