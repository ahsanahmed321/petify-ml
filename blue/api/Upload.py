import boto3
import os
session = boto3.session.Session()

dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
dirname = dirname + "/api/temp/"

client = session.client('s3',
                        region_name='sfo2',
                        endpoint_url='https://pmpai.sfo2.digitaloceanspaces.com',
                        aws_access_key_id='DMMTHE4OUKZ2XUYKHHIK',
                        aws_secret_access_key='pxfJEvq63XQ7jYuHAA9BS/QzkwLQae0aG8x9QA3545o')

print(dirname)
def upload_file(directory,cr_img):
    client.upload_file(dirname+cr_img,  # Path to local file
                    'pmpai',  # Name of Space
                     directory+'/'+cr_img,
                     ExtraArgs={'ACL': 'public-read'})  # Name for remote file
    os.remove(dirname+cr_img)                 
    print("File uploaded")                   

def upload_file_guest(directory,cr_img):
    client.upload_file(dirname+cr_img,  # Path to local file
                    'pmpai',  # Name of Space
                    'GUEST/'+directory+'/'+cr_img,
                    ExtraArgs={'ACL': 'public-read'})  # Name for remote file
    os.remove(dirname+cr_img)                 
    print("Guest File uploaded") 
