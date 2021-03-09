import boto3
import os
import cv2

dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
dirname = dirname + "/api/GID_IMAGES/"

def dosimage_guest(breed,filename):
    #labrador retriever,dog24
    print("Guest Dosimage")
    session = boto3.session.Session()
    client = session.resource('s3',
                            region_name='sfo2',
                            endpoint_url='https://pmpai.sfo2.digitaloceanspaces.com',
                            aws_access_key_id='DMMTHE4OUKZ2XUYKHHIK',
                            aws_secret_access_key='pxfJEvq63XQ7jYuHAA9BS/QzkwLQae0aG8x9QA3545o')

    bucket = client.Bucket('pmpai')
    print("guest session created")
    guest_list=[]
    cr_img = dirname+filename+'.jpg'
    #/GID_IMAGES/dog24.jpg
    print(cr_img)
    bucket.download_file('GUEST/'+breed+'/'+filename+'_crop.jpg', cr_img)  
    print("Downloaded")
    #GUEST/labrador_retriever/gid_crop.jpg,/api/GID_IMAGES/gid.jpg                    
    img = cv2.imread(cr_img)
    print("file added")
    guest_list.append(img)
    os.remove(cr_img)
    print("File removed")
    return guest_list

#av = dosimage("labrador_retriever",['2093cfdd4e64'])