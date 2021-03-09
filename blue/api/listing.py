import boto3 
#s3 = boto3.client("s3")
session = boto3.session.Session()
client = session.client('s3',
                        region_name='sfo2',
                        endpoint_url='https://pmpai.sfo2.digitaloceanspaces.com',
                        aws_access_key_id='DMMTHE4OUKZ2XUYKHHIK',
                        aws_secret_access_key='pxfJEvq63XQ7jYuHAA9BS/QzkwLQae0aG8x9QA3545o')


my_bucket = client.Bucket('pmpai')

for object_summary in my_bucket.objects.filter(Prefix="Tibetan spaniel/"):
    print(object_summary.key)

'''
all_objects = client.list_objects(Bucket = 'pmpai') 
print(all_objects)
'''