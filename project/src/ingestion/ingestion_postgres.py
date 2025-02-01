import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Define .env path
ENV_PATH = "../../.env"

# Load env file
load_dotenv(ENV_PATH)

# Define the env file
USER = os.getenv('POSTGRES_USER')
PASS = os.getenv('POSTGRES_PASSWORD')
HOST = os.getenv('POSTGRES_HOST')
PORT = os.getenv('POSTGRES_HOST_PORT')
DB = os.getenv('POSTGRES_MLFLOW_DB')

INGEST_DEST_FILE = "../../data/processed/car_postgres.csv"
INGEST_INDEX_LABEL = "index"
INGEST_SEP = "\t"

# Create connection
URL = f"postgresql://{USER}:{PASS}@{HOST}:{PORT}/{DB}"

# Create main function
def main():
    """
    This function used to ingest file from postgresql to local
    
    """
    try:
        
        # Connect to database
        engine = create_engine(URL)
        
        # Create query to get the data
        query = "select * from car"
        
        data = pd.read_sql_query(query, engine)
        
        data.to_csv(
            INGEST_DEST_FILE,
            sep = INGEST_SEP,
            index_label = INGEST_INDEX_LABEL
        )
        
        print('Ingesting data from postgresql success!')
    except Exception as e:
        print(str(e))
        
if __name__ == '__main__':
    main()