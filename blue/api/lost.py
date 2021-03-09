import json
import requests
from requests import Request, Session
from PIL import Image
import PIL
import numpy as np
from io import  BytesIO
import cv2
from urllib.request import urlopen

def lost_dog_list(lat,lng,breed):
        headers = {'Content-Type': 'multipart/form-data',
                'Accept': 'application/json'
                }
        files = {
                'server_key': 'f28ce8096b13cfc4e385a1ef396dd94e',
                'lat':lat,
                'lng':lng,
                'breed':breed
                }
        api_url = 'https://sandy.petmypal.biz/api/get-lost-pets'
        normal_multipart_req = Request('POST', api_url, data=files).prepare()
        request = normal_multipart_req
        s = Session()
        response = s.send(request)
        #print(response.status_code, response.text)
        loc = json.loads(response.text).get('pets')

        if len(loc) == 0:
                ret = [0,"No lost dog found"]
                return ret
        else:        
                try:
                        #print('Length of lost dogs is:',len(loc))
                        #print(response.text)
                        id_list=[]
                
                        #for x in range(0,len(loc)):
                        var = loc[0][0].get("total_pets")
                        print("Total pets: ",var)

                        lost = loc[0][0].get("lost_pets")
                        #print(los)
                        for x in range(var):
                                #print(x)
                                pet = lost[x]
                                y = pet.get('user')
                                key = y.get('pet_key')
                                print("Pet_key: ",key)
                                id_list.append(key)

                        ret = ["Not_Exception",id_list]
                        return ret
                except Exception as e:
                        ret = ["Exception",e]
                        return ret


#result = lost_dog_list(24.99597,67.059852,96)
#print(result)                