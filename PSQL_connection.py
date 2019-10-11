import psycopg2 as psql

def create_table():
    commands = (
        """
        CREATE TABLE psy (
            id SERIAL PRIMARY KEY,
            rasa VARCHAR(255) NOT NULL,
            wiek INT NOT NULL
        )
        """
    )

conn = None
try:


connection = psql.connect(
    database="test02",
    user="postgres",
    password="misiaczeko",
    host="localhost",
    port="5432"
)





