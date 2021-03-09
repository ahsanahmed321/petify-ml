import boto3
from routes import Register

session = boto3.session.Session()
client = session.client('s3',
                        region_name='sfo2',
                        endpoint_url='https://pmpai.sfo2.digitaloceanspaces.com',
                        aws_access_key_id='DMMTHE4OUKZ2XUYKHHIK',
                        aws_secret_access_key='pxfJEvq63XQ7jYuHAA9BS/QzkwLQae0aG8x9QA3545o')


def upload_file(directory,image):
    client.upload_file('temp/xyz.jpg',  # Path to local file
                    'pmpai',  # Name of Space
                    'Leonberger/file.jpg')  # Name for remote file
print("File uploaded")                   

