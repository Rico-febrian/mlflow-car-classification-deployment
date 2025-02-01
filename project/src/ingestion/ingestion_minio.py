import os
import urllib3
import copy
import pandas as pd
from dotenv import load_dotenv
from minio import Minio

# Define .env path
ENV_PATH = "../../.env"

# Load env file
load_dotenv(ENV_PATH)

# Disable warnings
urllib3.disable_warnings()

# Define the env file
ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
URL = os.getenv('MINIO_URL')
TLS = os.getenv('MINIO_TLS')

BUCKET_NAME = os.getenv('MINIO_BUCKET_NAME')
INGEST_SOURCE_FILE = "car.data"
INGEST_DEST_FILE = "../../data/processed/car_data.csv"
INGEST_INDEX_LABEL = "index"
INGEST_SEP = "\t"

# Create main function
def main():
    """
    This function used to ingest file from minio to local
    
    """

    # Create minio instance
    client = Minio(
            endpoint= URL,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=TLS,
            cert_check= (not TLS)
    )

    if(not client.bucket_exists(BUCKET_NAME)):
        print('Bucket not found!')
        return 0
    
    try:
        res = client.get_object(
            bucket_name = BUCKET_NAME,
            object_name = INGEST_SOURCE_FILE
        )

        data = copy.deepcopy(res.data.decode())
        
        data = pd.DataFrame([row.split(",") for row in data.splitlines()])
        
        data.to_csv(
            INGEST_DEST_FILE,
            sep = INGEST_SEP,
            index_label = INGEST_INDEX_LABEL
        )
        
        print('Fetching data success!')
    except Exception as e:
        print(str(e))
        
if __name__ == '__main__':
    main()