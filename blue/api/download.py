import boto3
import os
import cv2

dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
dirname = dirname + "/api/ID_IMAGES/"

def dosimage(breed,filename):
    print("Dosimage")
    session = boto3.session.Session()
    client = session.resource('s3',
                            region_name='sfo2',
                            endpoint_url='https://pmpai.sfo2.digitaloceanspaces.com',
                            aws_access_key_id='DMMTHE4OUKZ2XUYKHHIK',
                            aws_secret_access_key='pxfJEvq63XQ7jYuHAA9BS/QzkwLQae0aG8x9QA3545o')

    bucket = client.Bucket('pmpai')
    print("session created")
    avatar_list = []
    for each_file in filename:
        cr_img = dirname+each_file+'.jpg'
        print(cr_img)
        bucket.download_file(breed+'/'+each_file+'_crop.jpg', cr_img)  
        print("Downloaded")
        #labrador_retriever/puid_crop.jpg,/api/ID_IMAGES/puid.jpg                    
        img = cv2.imread(cr_img)
        print("file added")
        avatar_list.append(img)
        os.remove(cr_img)
        print("File removed")
    return avatar_list

#av = dosimage("labrador_retriever",['2093cfdd4e64'])