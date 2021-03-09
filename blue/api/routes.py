from flask import Blueprint,jsonify, request
from flask_restful import Api,Resource
import tensorflow as tf
from werkzeug.utils import secure_filename
from requests import Request, Session
import sys
import os
import json
import dlib
from imutils import face_utils
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
import PIL
import numpy as np
import cv2
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy_utils import ScalarListType
from sqlalchemy.exc import IntegrityError
import pickle
import gc
from requests_toolbelt.multipart.encoder import MultipartEncoder
import time
import validators
from threading import Thread
tf.keras.backend.clear_session()

dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
path = dirname + "/api"
sys.path.append(path)
detector_path = path + "/dogHeadDetector.dat"
predictor_path  = path + "/landmarkDetector.dat"
cropimage_path = path + "/temp/"
sys.path.append(path)
db_path = dirname + "/MissingDB"
sys.path.append(db_path)

from Functions import Main_Processing
from Functions import Main_Processing_For_Identification
from yolo_object_detection import yolo_return_names
from Features import Feature_Match
from Features import Detect_And_Crop_Face
from Features import Image_Matching
from Features import return_top3_score
from Landmarks import Update_Landmarks
from Landmarks import Landmarks_Calc
from Landmarks import string_to_list
from Upload import upload_file
from Upload import upload_file_guest
from lost import lost_dog_list
from download_guest import dosimage_guest
from download import dosimage
from breed import breed_processing


#from blue import app
import logging

import tracemalloc
tracemalloc.start()

mod = Blueprint('api',__name__)
api = Api(mod)

#database
db = SQLAlchemy()

logging.basicConfig(filename='demo.log',level=logging.DEBUG,
format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s")


detector = dlib.cnn_face_detection_model_v1(detector_path)
predictor = dlib.shape_predictor(predictor_path)

class_num = {'affenpinscher':1,#null
                'afghan_hound':2,
                'airedale_terrier':3,
                'akita':4,
                'alaskan_malamute':5,
                'american_eskimo_dog':6,#miss
                'american_foxhound':7,
                'american_staffordshire_terrier':8,
                'american_water_spaniel':9,
                'anatolian_shepherd_dog':10,
                'australian_cattle_dog':11,
                'australian_shepherd':12,
                'australian_terrier':13,
                'basenji':14,
                'basset_hound':15,
                'beagle':16,
                'bearded_collie':17,
                'beauceron':18,
                'bedlington_terrier':19,
                'belgian_malinois':20,
                'belgian_sheepdog':21,
                'belgian_tervuren':22,
                'bernese_mountain_dog':23,
                'bichon_frise':24,
                'black_and_tan_coonhound':25,
                'black_russian_terrier':26,
                'bloodhound':27,
                'bluetick_coonhound':28,#miss
                'border_collie':29,
                'border_terrier':30,
                'borzoi':31,
                'boston_terrier':32,
                'bouvier_des_flandres':33,
                'boxer':34,
                'boykin_spaniel':35,#miss
                'briard':36,
                'brittany':37,
                'brussels_griffon':38,
                'bull_terrier':39,
                'bulldog':40,
                'bullmastiff':41,
                'cairn_terrier':42,
                'canaan_dog':43,
                'cane_corso':44,#miss
                'cardigan_welsh_corgi':45,
                'cavalier_king_charles_spaniel':46,
                'chesapeake_bay_retriever':47,
                'chihuahua':48,
                'chinese_crested':49,
                'chinese_shar-pei':50,
                'chow_chow':51,
                'clumber_spaniel':52,
                'cocker_spaniel':53,
                'collie':54,
                'curly-coated_retriever':55,
                'dachshund':56,
                'dalmatian':57,
                'dandie_dinmont_terrier':58,
                'doberman_pinscher':59,
                'dogue_de_bordeaux':60,
                'english_cocker_spaniel':61,
                'english_setter':62,
                'english_springer_spaniel':63,
                'english_toy_spaniel':64,
                'entlebucher_mountain_dog':65,#miss
                'field_spaniel':66,
                'finnish_spitz':67,
                'flat-coated_retriever':68,
                'french_bulldog':69,
                'german_pinscher':70,
                'german_shepherd_dog':71,
                'german_shorthaired_pointer':72,
                'german_wirehaired_pointer':73,
                'giant_schnauzer':74,
                'glen_of_imaal_terrier':75,
                'golden_retriever':76,
                'gordon_setter':77,
                'great_dane':78,
                'great_pyrenees':79,
                'greater_swiss_mountain_dog':80,
                'greyhound':81,
                'havanese':82,
                'ibizan_hound':83,
                'icelandic_sheepdog':84,#miss
                'irish_red_and_white_setter':85,#miss
                'irish_setter':86,
                'irish_terrier':87,
                'irish_water_spaniel':88,
                'irish_wolfhound':89,
                'italian_greyhound':90,
                'japanese_chin':91,
                'keeshond':92,
                'kerry_blue_terrier':93,
                'komondor':94,
                'kuvasz':95,
                'labrador_retriever':96,
                'lakeland_terrier':97,
                'leonberger':98,#miss
                'lhasa_apso':99,
                'lowchen':100,
                'maltese':101,#miss
                'manchester_terrier':102,#miss
                'mastiff':103,
                'miniature_schnauzer':104,
                'neapolitan_mastiff':105,
                'newfoundland':106,
                'norfolk_terrier':107,
                'norwegian_buhund':108,#miss 
                'norwegian_elkhound':109,
                'norwegian_lundehund':110,#miss
                'norwich_terrier':111,
                'nova_scotia_duck_tolling_retriever':112,#null 
                'old_english_sheepdog':113,#miss
                'otterhound':114,
                'papillon':115,
                'parson_russell_terrier':116,
                'pekingese':117,
                'pembroke_welsh_corgi':118,
                'petit_basset_griffon_vendeen':119,
                'pharaoh_hound':120,
                'plott':121,
                'pointer':122,
                'pomeranian':123,
                'poodle':124,#miss
                'portuguese_water_dog':125,
                'saint_bernard':126,
                'silky_terrier':127,
                'smooth_fox_terrier':128,
                'tibetan_mastiff':129,
                'welsh_springer_spaniel':130,
                'wirehaired_pointing_griffon':131,#d
                'xoloitzcuintli':132,#miss
                'yorkshire_terrier':133,
                'shetland_sheepdog':134,
                'english_foxhound':135,
                'african_hunting_dog':136,
                'dhole':137,
                'dingo':138,
                'mexican_hairless':139,
                'standard_poodle':140,#d
                'miniature_poodle':141,
                'toy_poodle':142,
                'brabancon_griffon':143,
                'samoyed':144,
                'pug':145,
                'malamute':146,
                'eskimo_dog':147,
                'entleBucher':148,
                'appenzeller':149,
                'miniature_pinscher':150,
                'rottweiler':151,
                'kelpie':152,
                'malinois':153,
                'groenendael':154,
                'schipperke':155,
                'siberian_husky':156,
                'sussex_spaniel':157,
                'vizsla':158,
                'west_Highland_white_terrier':159,
                'scotch_terrier':160,
                'sealyham_terrier':161,
                'irish_terrier':162,
                'shih-Tzu':163,
                'japanese_spaniel':164,
                'redbone':165,
                'walker_hound':166,
                'wire-haired_fox_terrier':167,
                'whippet':168,
                'weimaraner':169,
                'soft-coated_wheaten_terrier':170,
                'staffordshire_bullterrier':171,
                'scottish_deerhound':172,
                'saluki':173,
                'blenheim_spaniel':174,
                'toy_terrier':175,
                'rhodesian_ridgeback':176,
                'standard_schnauzer':177,
                'tibetan_terrier':178,
                'miniature poodle':179,
                'harrier':180,
                'jack russel terrier':181,
                'polish lowland sheepdog':182,
                'dogo argentino':183,
                'miniature bull terrie':184,
                'miniature american eskimo dog':185,
                'puli':186,
                'shiba inu':187,
                'skye terrier':188,
                'spinone italiano':189,
                'swedish vallhund':190,
                'tibetan spaniel':191,
                'toy fox terrier':192,
                'toy manchester terrier':193,
                'welsh terrier':194
}

# class Dog_Breeds(Resource):
#     def post(self):
       
#         try:
#             ##tf.keras.backend.clear_session()
            
#             #app.logger.info('Processing default request')
#             image = request.files['image'].read()

#             l=yolo_return_names(image)
#             if "dog" in l:

#                 print("-----------------------HITT------------------")
#                 # ... run your application ...

#                 ret=Main_Processing(image)

#                 snapshot = tracemalloc.take_snapshot()
#                 top_stats = snapshot.statistics('lineno')
#                 print("[ Top 15 ]")
#                 for stat in top_stats[:15]:
#                     print(stat)
                
#                 tf.keras.backend.clear_session()
#                 gc.collect()
#                 return jsonify(ret)
#             else:
#                 tf.keras.backend.clear_session()
#                 gc.collect()
#                 return jsonify({"api_status":400,"msg":"Dog image is not present"})
#         except Exception as e:
#             return jsonify({"api_status":400,"msg":"Can not handle this image","problem":e})

class Register(Resource):
    def post(self):
        try:
            #app.logger.info('Processing default request')
            print("-----------------------HITT------------------")

            
        
            image_url=postedData['image_url']
            PUID=postedData['puid']
            breed=postedData['breed']
            crop_image_url=postedData['crop_image_url']

            #image_url = 'https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg'
            valid=validators.url(image_url)
            if valid==True:

                print(image_url)
                print(PUID)
                print(breed)
                print(crop_image_url)

                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                
                imgarr = np.array(img) 
                print(imgarr)

                if breed == None or breed == "":
                    #Find Breed
                    dic_result=Main_Processing_For_Identification(imgarr)
                    if dic_result[0] == "Not_Exception":
                        dic = dic_result[1]
                        print(dic)
                        breed_list_add = dic["breed"]
                        print("breed list: ",breed_list_add)

                        breed_list=string_to_list(breed_list_add)

                        find_breed = breed_list[0]
                        print("breed:",find_breed)
                    else:
                        er = dic_result[1]
                        ret = {"api_status":400,"msg":"can not handle this image","Problem":er}
                        return jsonify(ret)
                else:
                    #Find Breed
                    find_breed = breed
                    print("breed:",find_breed)

                try:
                    if crop_image_url == None or crop_image_url == "":
                        #crop image
                        print("Crop Image is not present");print("Croping Image and Calculating landmarks")

                        my_list = Detect_And_Crop_Face(imgarr,detector)
                        status_img=my_list[0]
                        img_crop=my_list[1]
                
                        cr_name=str(PUID)+'_crop.jpg' 
                        print(cropimage_path+cr_name)
                        
                        
                        if status_img == "Image":
                            #Find Landmarks
                            cv2.imwrite(os.path.join(cropimage_path,str(PUID)+'_crop.jpg'),img_crop) 
                            equation = Landmarks_Calc(img_crop,detector,predictor)
                            print("Equation Result: ",equation)   
                            

                            #db    
                            us=user(id=PUID,eq=equation)
                            db.session.add(us)
                            db.session.commit()

                            print(find_breed,cr_name)
                            upload_file(find_breed,cr_name)
                            return jsonify({"api_status":200,"msg":"Successfully Registered Pet"})    
                
                        else:
                            dictionary =  {"msg": "Can't find proper face angle of the dog, Click and try again."}
                            print("Dictinary Output: ",dictionary)
                            return dictionary
                    
                    else:
                        #crop image present find landmarks
                        print("Crop Image is present");print("Calculating landmarks")
                        crop_img_url = crop_image_url
                    
                        response = requests.get(crop_img_url)
                        cr_img = Image.open(BytesIO(response.content))
                        cr_arr = np.array(cr_img) 
                        print(cr_arr)
                        
                        cr_name=str(PUID)+'_crop.jpg' 
                        print(cropimage_path+cr_name)
                        cv2.imwrite(os.path.join(cropimage_path,str(PUID)+'_crop.jpg'),cr_arr) 

                        equation = Landmarks_Calc(cr_img,detector,predictor)
                        print("Equation Result: ",equation)

                                    
                        #db    
                        us=user(id=PUID,eq=equation)
                        db.session.add(us)
                        db.session.commit()
                        
                        upload_file(find_breed,cr_name)

                        return jsonify({"api_status":200,"msg":"Successfully Registered Pet"})
                except IntegrityError as e:
                        print("Unique key Error")
                        #ret = {"api_status":400,"msg":"Registration ID must be unique","problem":e}
                        #return ret
                        return jsonify(api_status=400,msg="Registration ID must be unique",problem=str(e))
            if valid != True:
                ret = {"api_status":400,"msg":"Enter Valid URL","problem":"Invalid URL"}
                return jsonify(ret)

                 
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            print("[ Top 15 ]")
            for stat in top_stats[:15]:
                print(stat)
            
            tf.keras.backend.clear_session()
            gc.collect()

            
        except Exception as e:
            ret = {"api_status":301,"msg":"Unsuccessful Registration","problem":e}
            return jsonify(ret)


class Guest_find(Resource):
    def post(self):
        try:
            #app.logger.info('Processing default request')
            print("-----------------------HITT------------------")
            print("guest API")

            #image = request.files['image'].read()
            #print(image)
            guest_id = request.form['GID']
            image_url = request.form['image_url']
            # loc = request.form['location']

            # loc = json.loads(loc)

            # latitude = loc.get('lat')
            # longitude = loc.get('long')
            valid=validators.url(image_url)
            if valid==True:

                print(guest_id)
                print(image_url)
                #print(loc)
                
                retur = breed_processing(image_url)

                if retur[0] == "Not_Exception":
                    ret = retur[1]
                    breeds = ret["breed"]
                    breed_list=string_to_list(breeds)
                    top_breed = breed_list[0]

                    #npimg = np.fromstring(image, np.uint8)
                    #img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
                    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                    response = requests.get(image_url)
                    print(response)
                    img = Image.open(BytesIO(response.content))
                    print(img)
                    
                    imgarr = np.array(img) 
                    print(imgarr)


                    my_list = Detect_And_Crop_Face(imgarr,detector)
                    status_img=my_list[0]
                    img_crop=my_list[1]
                    cr_name=guest_id+'_crop.jpg'
                    print(cropimage_path+guest_id+'_crop.jpg')

                    if status_img == "Image":
                        #Find Landmarks
                        cv2.imwrite(os.path.join(cropimage_path,str(guest_id)+'_crop.jpg'),img_crop)
                       
                        equation = Landmarks_Calc(img_crop,detector,predictor)
                        print("Equation Result: ",equation)

                        #db    
                        print("Inserting in DB")

                
                        try:    
                            print("guest DB")            
                            #gu=guest(gid=guest_id,geq=equation,brd=top_breed,glog=longitude,glat=latitude)
                            gu=guest(gid=guest_id,geq=equation,brd=top_breed)
                            db.session.add(gu)
                            db.session.commit() 
                            print("DB done")
                            upload_file_guest(top_breed,cr_name)
                            return jsonify({"api_status":200,"msg":"Successfully Save Guest"})

                        except IntegrityError as e:
                            print("Unique key Error")
                            sqlerr = {"api_status":400,"msg":"Guest ID must be unique","problem":str(e)}
                            return sqlerr
                            #print(e)
                            #return jsonify(api_status=400,msg="Guest ID must be unique",problem=str(e))
    

                    else:
                        dictionary =  {"msg": "Can't find proper face angle of the dog, Click and try again."}
                        print("Dictinary Output: ",dictionary)
                        return dictionary

                else:
                    er = retur[1]
                    ret = {"api_status":400,"msg":"Cannot find breed","problem":er}
                    return jsonify(ret)
            if valid != True:
                ret = {"api_status":400,"msg":"Enter Valid URL","problem":"Invalid URL"}
                return jsonify(ret)
               
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            print("[ Top 15 ]")
            for stat in top_stats[:15]:
                print(stat)
            tf.keras.backend.clear_session()
            gc.collect()

        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)

class Guest_Matching(Resource):
    def post(self):
        try:
            print("----------------------HITT------------------")
            #postedData=request.get_json()
            #print(postedData)
        
            #guest_id=postedData['GID']
            #loc=postedData['location']
            #breed=postedData['breed']

            guest_id = request.form['GID']
            loc = request.form['location']
            
            loc = json.loads(loc)

            latitude = loc.get('lat')
            longitude = loc.get('long')

            try:
                #guest eq
                eq = guest.query.with_entities(guest.geq).filter(guest.gid==guest_id).first()
                equation = eq[0]
                print("Guest equation: ",equation)

                #guest breed
                brd = guest.query.with_entities(guest.brd).filter(guest.gid==guest_id).first()
                breed = brd[0]
                print("Guest breed: ",breed)

                #breed_dict=pickle.load(open("breed_dict.p","rb"))
                
                breed_no = class_num.get(breed)
                        
                lost = lost_dog_list(latitude,longitude,breed_no)
                
                try:
                    if lost[0] == "Not_Exception":
                        lost_keys = lost[1]
                        
                        print("latitude: ",latitude)
                        print("longitude: ",longitude)
                        print("Lost keys: ",lost_keys)
                            
                        landmarks = []
                        print("OUT_LOOP")
                        for key in lost_keys:
                            print("IN_LOOP")
                            record = user.query.with_entities(user.eq).filter(user.id==key).first()
                            print(record)
                            recordx=record[0]
                            print(recordx)
                            landmarks.append(recordx)
                        print(landmarks)    

                        lost_avatar_list = dosimage(breed,lost_keys)

                        l = {}
                        lenth = len(lost_keys)
                        for i in range(0,lenth):
                            l[lost_keys[i]]=lost_avatar_list[i]

                        print("Lost Dictionary: ",l)
                        #guest dos image
                        guest_img = dosimage_guest(breed,guest_id)


                        data_frame =  Image_Matching(guest_img,equation,lost_avatar_list,landmarks,detector,predictor)
                        print(data_frame)
                            

                        var = return_top3_score(l,data_frame)
                        p_list = var['Top Match'] 
                        puid = []
                        target=[]
                        print(p_list)
                        for p in p_list:
                            pid = p['PUID']
                            pt = p['Target']
                            puid.append(pid)
                            target.append(pt) 
                        print(puid)
                        print(target)

                        
                        for each in range(len(puid)):
                            each_row = lost_matching.query.with_entities(lost_matching.reg_id).filter(lost_matching.reg_id==puid[each]).first()
                            #register id
                            print(each_row)

                            
                            if each_row == None or each_row == "":
                                g=[]
                                t=[]
                                #g.append(guest_id[1:-1])
                                g.append(guest_id)
                                print(target[each])
                                t.append(target[each])
                                lost=lost_matching(reg_id=puid[each],guest_id=g,guest_target=t)
                                db.session.add(lost)
                                db.session.commit()
                                print("New record added")
                                

                            else:
                                each_guest = lost_matching.query.with_entities(lost_matching.guest_id).filter(lost_matching.reg_id==puid[each]).first()
                                #guest id
                                x = each_guest[0]
                                x.append(guest_id)
                                print("Guest ID List: ",x)

                                each_target = lost_matching.query.with_entities(lost_matching.guest_target).filter(lost_matching.reg_id==puid[each]).first()
                                #guest target
                                print(each_target)
                                y = each_target[0]
                                print(y)
                                y.append(target[each])
                                print("Guest Target List: ",y)

                                r = lost_matching.query.filter(lost_matching.reg_id==puid[each]).delete()
                                db.session.commit()
                                print("Remove previous record")

                                lt=lost_matching(reg_id=puid[each],guest_id=x,guest_target=y)
                                db.session.add(lt)
                                db.session.commit() 
                                print("Previous record updated")
                                
    
                        print("Done")
                        return jsonify({"api_status":200,"msg":"Successfully Save Guest"})

                    else:
                        er = lost[1]
                        ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":er}
                        return jsonify(ret)
    

                except Exception as e:
                    print("Registered key not found")
                    regerr = {"api_status":400,"msg":"Key Error","problem":str(e)}
                    return regerr          
                   
            except Exception as e:
                    print("Guest key not found")
                    regerr = {"api_status":400,"msg":"Can't find data of this GID in AI_GUEST_DB","problem":str(e)}
                    return regerr
                                

            
                #return jsonify(api_status=400,msg="Can't find data of this PUID in AI_PET_DB",problem=str(e)) 

        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)    

class Guest_IDs(Resource):
    def post(self):
        try:
            print("----------------------HITT------------------")

            #Data=request.get_json()
            #print(Data)
        
            #pet_id=Data['puid']
            pet_id = request.form['puid']
            print(pet_id)

            result = lost_matching.query.with_entities(lost_matching.reg_id,lost_matching.guest_id,lost_matching.guest_target).filter(lost_matching.reg_id==pet_id).first()
            print("Guest Dog List",result)
            ids={}
            prob={}           

        
            if result != None:
                guest_ids=result[1]
                guest_targets=result[2]
                for i in range(len(guest_ids)):
                    prob[guest_ids[i]]=guest_targets[i]
 
                ids[result[0]]=prob
                print(ids) 
                
                return jsonify(ids)
            else:
                err = {"api_status":400,"msg":"Record not found"}
                return jsonify(err)

        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)

class Breed_Dict(Resource):
    def get(self):
        try:
            print("-----------------------HITT------------------")
            print(class_num)   

            pickle.dump(class_num, open("Breed_dict.p", "wb")) 

            return jsonify(class_num)
        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)

class Lost_list(Resource):
    def get(self):
        try:
            print("-----------------------HITT------------------")
            
            rows = lost_matching.query.with_entities(lost_matching.reg_id, lost_matching.guest_id,lost_matching.guest_target).all()
            print("Record: ",rows)
            record={}

            for each_record in rows:
                print("Each record: ",each_record)
                guest_ids = each_record[1]
                print("Guest ID: ",guest_ids)
                guest_target = each_record[2]
                print("Guest Target",guest_target)
                new_list=[]
                full_list=[]
                for x in range(len(guest_ids)):
                    print(x)
                    new_list=[guest_ids[x],guest_target[x]]
                    full_list.append(new_list)
                    print("Combine List: ",new_list)
                print("Full List: ",full_list)
                record[each_record[0]]=full_list
                print("record: ",record)

            print(record)    

            return jsonify(record)

        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)


class user(db.Model):
    __tablename__ = "AI_Pet_DB"
    id = db.Column(db.String(50), primary_key=True)
    eq = db.Column(db.Float())

    def __repr__(self):
        return('REGISTERED PET '+str(self.id)+' '+str(self.eq))

class guest(db.Model):
    __tablename__ = "AI_Guest_DB"
    gid = db.Column(db.String(50), primary_key=True)
    geq = db.Column(db.Float())
    brd = db.Column(db.String(50))
    #glog = db.Column(db.Float())
    #glat = db.Column(db.Float())
    
    def __repr__(self):
        #return('GUEST PET'+str(self.gid)+' '+str(self.geq)+' '+str(self.brd))
        return('GUEST PET'+str(self.gid)+' '+str(self.geq))

class lost_matching(db.Model):
    __tablename__ = "Lost_Matching_DB"
    reg_id = db.Column(db.String(50), primary_key=True)
    guest_id = db.Column(ScalarListType())
    guest_target = db.Column(ScalarListType())

    def __repr__(self):
        return('LOST PET '+str(self.reg_id)+' '+str(self.guest_id)+' '+str(self.guest_target))

class Extract_Features(Resource):
    def post(self):
        try:
            #app.logger.info('Processing default request')
            image = request.files['image'].read()
            # breed_list = request.form['breeds']
            # print(breed_list)
            
            print("-----------------------HITT------------------")
            # ... run your application ...

            ret= Feature_Match(image,db_path,detector,predictor)

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            print("[ Top 15 ]")
            for stat in top_stats[:15]:
                print(stat)
            tf.keras.backend.clear_session()
            gc.collect()
            return jsonify(ret)

        except Exception as e:
            ret = {"status":301,"msg":"Picture quality is not good","problem":e}
            gc.collect()
            return jsonify(ret)


class Add_Dog_In_DB(Resource):
    def post(self):
        try:
            #app.logger.info('Processing default request')
            image = request.files.get('image', '')
            print("type",type(image))
            filename = image.filename
            #breed = request.form['breed']
            #dog_id = request.form['dog_id']
            
            #filename = image.filename
            # breed_path = db_path + "/" + breed
            # if not os.path.exists(breed_path):
            #     os.makedirs(breed_path)

            try:
                print("-----------------------HITT------------------")
                # ... run your application ...

                #msg = Update_Landmarks(image,breed,filename,db_path,detector,predictor)
                msg = Update_Landmarks(image,filename,db_path,detector,predictor)
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                print("[ Top 15 ]")
                for stat in top_stats[:15]:
                    print(stat)
                tf.keras.backend.clear_session()
                gc.collect()

                if msg == "Successfully Updated":
                    status = 200
                else:
                    status = 301

                ret = {"status":status,"msg":msg}
                return jsonify(ret)
            
            except Exception as e:
                ret = {"status":301,"msg":"Picture quality is not good","problem":e}
                return jsonify(ret)
        except Exception as e:
            ret = {"status":301,"msg":"Cannot add this image in DB","problem":e}
            return jsonify(ret)


#api.add_resource(Dog_Breeds,"/dog_breeds")
api.add_resource(Register,"/register")
api.add_resource(Guest_find,"/guest_dog")
api.add_resource(Breed_Dict,"/breed_data")
api.add_resource(Lost_list,"/lost_list")
api.add_resource(Guest_IDs,"/guest_id")
api.add_resource(Guest_Matching,"/guest_matching")

#api.add_resource(Dog_check,"/check")
#api.add_resource(Hit_Web,"/hit")
#api.add_resource(Hit_Web_Again,"/hit_again")
#api.add_resource(Breed,"/dog_seq")
# api.add_resource(Extract_Features,"/features")
# api.add_resource(Add_Dog_In_DB,"/upload")

