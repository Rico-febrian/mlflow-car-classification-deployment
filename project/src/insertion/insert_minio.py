import os
import urllib3
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
SOURCE_FILE = os.getenv('MINIO_SOURCE_FILE')
DEST_FILE = os.getenv('MINIO_DEST_FILE')

# Create main function
def main():
    """
    This function used to upload file to minio buckets
    
    """

    # Create minio instance
    client = Minio(
            endpoint= URL,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=TLS,
            cert_check= (not TLS)
    )
    
    found = client.bucket_exists(BUCKET_NAME)
    if(not found):
        print('Bucket not found!')
        return
    
    try:
        client.fput_object(
            BUCKET_NAME,
            DEST_FILE,
            SOURCE_FILE
        )
        
        print('File successfully uploaded!')
    except Exception as e:
        print(str(e))
        
if __name__ == '__main__':
    main()