from sqlalchemy import create_engine, text
import dotenv
import os

class Connector:
    def __init__(self, path: str = None):
        self.engine = None
        if not path:
            path = os.path.abspath(__file__).replace("connect_sql.py", ".env")
            print(f"No path provided, using default: {path}")
        self.path = path
        self.get_passwords()
    
    def get_passwords(self):
        dotenv.load_dotenv(self.path)
        env = dotenv.dotenv_values(self.path)
        self.connection_uri = f"postgresql+psycopg2://{env['DB_USER']}:{env['DB_PASSWORD']}@{env['DB_HOST']}:{env['DB_PORT']}/{env['DB_NAME']}?options=-c search_path={env['SCHEMA_NAME']}"
        self.engine = create_engine(self.connection_uri)

    def get_query(self, query: str):
        if self.engine is None:
            return "No engine found. Please check the connection."

        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
            return result
        return "There was an error executing the query."      

if __name__ == "__main__":
    connector = Connector()
    query = "SELECT t1.full_name, count(p.title) FROM people as t1 join projects as p on t1.unics_id = p.principal_investigator group by t1.full_name HAVING count(p.title) > 1"
    result = connector.get_query(query)
    for row in result:
        print("Row:", row)