import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from imutils import face_utils
from collections import Counter
from io import BytesIO
from PIL import Image
import requests

dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-2]
dirname = "/".join(dirname_list)
landmark_path = dirname + "/Landmarks_Data.xlsx"

from Functions import Main_Processing_For_Identification

landmark_df = pd.read_excel(landmark_path)
landmark_df = landmark_df.drop("Unnamed: 0",axis=1)
landmark_df = landmark_df[["Image2_Path","Landmarks_Missing_Image"]]

def landmark_shapes(img,detector,predictor):
    dets = detector(img, upsample_num_times=1)


    img_result = img.copy()
    shapes = []


    for i, d in enumerate(dets):
        shape = predictor(img, d.rect)
        shape = face_utils.shape_to_np(shape)
        
        for i, p in enumerate(shape):
            shapes.append(shape)
            break    
        break
    return shapes


def Eucliden_Formula(arr1,arr2):
    x1 = arr1[0]
    y1 = arr1[1]
    x2 = arr2[0]
    y2 = arr2[1]

    form = (x1 - x2)**2 + (y1 - y2)**2
    return form

def Mid_Point(arr1,arr4):
    x1 = arr1[0]
    y1 = arr1[1]
    x2 = arr4[0]
    y2 = arr4[1]

    midx = (x1 + x2)/2 
    midy = (y1 + y2)/2 

    mid = np.array([midx,midy])
    return mid

def midpoint_difference(mid,arr0):
    x1 = mid[0]
    y1 = mid[1]
    x2 = arr0[0]
    y2 = arr0[1]

    diff = ( x1 - x2 ) + (y1 - y2)
    return diff

def Trio(arr2,arr5,arr3):

    A = Eucliden_Formula(arr2,arr5)
    B = Eucliden_Formula(arr5,arr3)
    C = Eucliden_Formula(arr3,arr2)

    trio = A + B + C
    return trio

def Hexa(arr0,arr1,arr2,arr3,arr4,arr5):
    A = Eucliden_Formula(arr0,arr1)
    B = Eucliden_Formula(arr1,arr2)
    C = Eucliden_Formula(arr2,arr3)
    D = Eucliden_Formula(arr3,arr5)
    E = Eucliden_Formula(arr5,arr4)
    F = Eucliden_Formula(arr4,arr0)

    hexa_out = A + B + C + D + E + F
    return hexa_out

def Linear_Eqn(img,detector,predictor):
    alpha = 0.00001
    shapes = landmark_shapes(img,detector,predictor)
    print(shapes)
    shapes = shapes[0]
    Zero = shapes[0]
    One = shapes[1]
    Two = shapes[2]
    Three = shapes[3]
    Four = shapes[4]
    Five = shapes[5]
    
    A = Eucliden_Formula(Two,Five)
    B_1 = Mid_Point(One,Four)
    B = midpoint_difference(B_1,Zero)
    C = Eucliden_Formula(Zero,Three)
    trio = Trio(Two,Five,Three)
    hexa = Hexa(Zero,One,Two,Three,Four,Five)

 
    lin_eq = (alpha * A) + (alpha * B) + (alpha * C) + (alpha * trio) + (alpha * hexa)
    return lin_eq 


def Image_Matching(image,image_landmarks,images_list,images_landmarks_list,detector,predictor):
    data = pd.DataFrame(columns=["Image2_Path","Image_One_KeyPoints","Image_Two_KeyPoints",
    "Count_Matches","Landmarks_Input_Image","Landmarks_Missing_Image","Feature_Percentage",
    "Landmark_Difference","Percentage_Landmark","Target"])
    print("Image_Matching")
    print("Image List:",images_list)
    image_one = image
    print("Image one: ",image_one)
    Landmarks_Input_Image = image_landmarks
    for index in range(len(images_list)):
        image_two = images_list[index]
        print("Image two: ",image_two)

        #npimg = np.fromstring(image_two, np.uint8)
        #img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        #print("negative: ",img)
        #response = requests.get(image_one)
        #img1 = Image.open(BytesIO(response.content))
        #img1 = np.array(img1)
        #print("img",img)
        print("Before Features Extraction")

        count_match,len_keypoints_1, len_keypoints_2 =Feature_Extraction(image_one,image_two)
        print("After Features Extraction")
        data.loc[index,"Image2_Path"] = image_two
        data.loc[index,"Image_One_KeyPoints"] = len_keypoints_1
        data.loc[index,"Image_Two_KeyPoints"] = len_keypoints_2
        data.loc[index,"Count_Matches"] = count_match
        data.loc[index,"Landmarks_Input_Image"] = Landmarks_Input_Image
        landmark_value = images_landmarks_list[index]
        data.loc[index,"Landmarks_Missing_Image"] = landmark_value
        perc = calc_percentage_landmarks(len_keypoints_1,count_match)
        data.loc[index,"Feature_Percentage"] = perc

        minus = Difference_Landmarks(Landmarks_Input_Image,landmark_value)
        data.loc[index,"Landmark_Difference"] = minus
        land_per = Landmark_Percentage(minus)
        data.loc[index,"Percentage_Landmark"] = land_per
        target_sum = Target_Sum(perc,land_per)
        data.loc[index,"Target"] = target_sum
        print("Image one keypoints: ",len_keypoints_1)
        print("Image two keypoints: ",len_keypoints_2)
        print("Count match",count_match)
        print("Landmarks input image: ",Landmarks_Input_Image)
        print("Landmarks value: ",landmark_value)
        print("Percentage: ",perc)
        print("Minus: ",minus)
        print("Percentage_Landmark: ",land_per)
        print("Target: ",target_sum)
        print("Data:",data)
        print("Index:",index)
    return data

def Landmarks_Calc(img,detector,predictor):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    equation = Linear_Eqn(img,detector,predictor)
    return equation



def Feature_Extraction(imgOne,imgTwo):
    print("Feature Extraction")
    imgOne = imgOne[0]
    img1 = cv2.cvtColor(imgOne, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(imgTwo, cv2.COLOR_BGR2GRAY)

    ##### sift
    print("Sift")
    sift = cv2.xfeatures2d.SIFT_create()

    print("keypoints1")
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    print("keypoints2")
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)


    #feature matching

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    count_match =  len(matches)
    print("Complete Feature Extraction")
    return count_match,len(keypoints_1), len(keypoints_2)

def calc_percentage_landmarks(num1,num2):
    perc = (num2 / num1) * 100
    return perc

def Difference_Landmarks(num1,num2):
    res = 0
    if num1 >= num2:
        res = num1 - num2
    elif num2 > num1:
        res = num2 - num1
    return res

def Landmark_Percentage(num1):
    if num1 < 1:
        res = 1 - num1
        res = res * 100
        return res
    else:
         return 0

def Target_Sum(num1,num2):
    res = num1 + num2
    res = res / 2
    return res

def Match_in_DB(image,db_path,breed_list,detector,predictor):
    
    landmark_df = pd.read_excel(landmark_path)
    landmark_df = landmark_df.drop("Unnamed: 0",axis=1)
    landmark_df = landmark_df[["Image2_Path","Landmarks_Missing_Image","Breeds"]]
    
    data = pd.DataFrame(columns=["Image2_Path","Image_One_KeyPoints","Image_Two_KeyPoints","Count_Matches","Landmarks_Input_Image","Landmarks_Missing_Image","Feature_Percentage","Landmark_Difference","Percentage_Landmark","Target"])
    image_one = image
    DATA_DIR = db_path
    Breed_lst = breed_list
    CATEGORIES = ['affenpinscher',
                'afghan_hound',
                'airedale_terrier',
                'akita',
                'alaskan_malamute',
                'american_eskimo_dog',
                'american_foxhound',
                'american_staffordshire_terrier',
                'american_water_spaniel',
                'anatolian_shepherd_dog',
                'australian_cattle_dog',
                'australian_shepherd',
                'australian_terrier',
                'basenji',
                'basset_hound',
                'beagle',
                'bearded_collie',
                'beauceron',
                'bedlington_terrier',
                'belgian_malinois',
                'belgian_sheepdog',
                'belgian_tervuren',
                'bernese_mountain_dog',
                'bichon_frise',
                'black_and_tan_coonhound',
                'black_russian_terrier',
                'bloodhound',
                'bluetick_coonhound',
                'border_collie',
                'border_terrier',
                'borzoi',
                'boston_terrier',
                'bouvier_des_flandres',
                'boxer',
                'boykin_spaniel',
                'briard',
                'brittany',
                'brussels_griffon',
                'bull_terrier',
                'bulldog',
                'bullmastiff',
                'cairn_terrier',
                'canaan_dog',
                'cane_corso',
                'cardigan_welsh_corgi',
                'cavalier_king_charles_spaniel',
                'chesapeake_bay_retriever',
                'chihuahua',
                'chinese_crested',
                'chinese_shar-pei',
                'chow_chow',
                'clumber_spaniel',
                'cocker_spaniel',
                'collie',
                'curly-coated_retriever',
                'dachshund',
                'dalmatian',
                'dandie_dinmont_terrier',
                'doberman_pinscher',
                'dogue_de_bordeaux',
                'english_cocker_spaniel',
                'english_setter',
                'english_springer_spaniel',
                'english_toy_spaniel',
                'entlebucher_mountain_dog',
                'field_spaniel',
                'finnish_spitz',
                'flat-coated_retriever',
                'french_bulldog',
                'german_pinscher',
                'german_shepherd_dog',
                'german_shorthaired_pointer',
                'german_wirehaired_pointer',
                'giant_schnauzer',
                'glen_of_imaal_terrier',
                'golden_retriever',
                'gordon_setter',
                'great_dane',
                'great_pyrenees',
                'greater_swiss_mountain_dog',
                'greyhound',
                'havanese',
                'ibizan_hound',
                'icelandic_sheepdog',
                'irish_red_and_white_setter',
                'irish_setter',
                'irish_terrier',
                'irish_water_spaniel',
                'irish_wolfhound',
                'italian_greyhound',
                'japanese_chin',
                'keeshond',
                'kerry_blue_terrier',
                'komondor',
                'kuvasz',
                'labrador_retriever',
                'lakeland_terrier',
                'leonberger',
                'lhasa_apso',
                'lowchen',
                'maltese',
                'manchester_terrier',
                'mastiff',
                'miniature_schnauzer',
                'neapolitan_mastiff',
                'newfoundland',
                'norfolk_terrier',
                'norwegian_buhund',
                'norwegian_elkhound',
                'norwegian_lundehund',
                'norwich_terrier',
                'nova_scotia_duck_tolling_retriever',
                'old_english_sheepdog',
                'otterhound',
                'papillon',
                'parson_russell_terrier',
                'pekingese',
                'pembroke_welsh_corgi',
                'petit_basset_griffon_vendeen',
                'pharaoh_hound',
                'plott',
                'pointer',
                'pomeranian',
                'poodle',
                'portuguese_water_dog',
                'saint_bernard',
                'silky_terrier',
                'smooth_fox_terrier',
                'tibetan_mastiff',
                'welsh_springer_spaniel',
                'wirehaired_pointing_griffon',
                'xoloitzcuintli',
                'yorkshire_terrier',
                'shetland_sheepdog',
                'english_foxhound',
                'african_hunting_dog',
                'dhole',
                'dingo',
                'mexican_hairless',
                'standard_poodle',
                'miniature_poodle',
                'toy_poodle',
                'brabancon_griffon',
                'samoyed',
                'pug',
                'malamute',
                'eskimo_dog',
                'entleBucher',
                'appenzeller',
                'miniature_pinscher',
                'rottweiler',
                'kelpie',
                'malinois',
                'groenendael',
                'schipperke',
                'siberian_husky',
                'sussex_spaniel',
                'vizsla',
                'west_Highland_white_terrier',
                'scotch_terrier',
                'sealyham_terrier',
                'irish_terrier',
                'shih-Tzu',
                'japanese_spaniel',
                'redbone',
                'walker_hound',
                'wire-haired_fox_terrier',
                'whippet',
                'weimaraner',
                'soft-coated_wheaten_terrier',
                'staffordshire_bullterrier',
                'scottish_deerhound',
                'saluki',
                'blenheim_spaniel',
                'toy_terrier',
                'rhodesian_ridgeback',
                'standard_schnauzer',
                'tibetan_terrier',
                'miniature poodle',
                'harrier',
                'jack russel terrier',
                'polish lowland sheepdog',
                'dogo argentino',
                'miniature bull terrie',
                'miniature american eskimo dog',
                'puli',
                'shiba inu',
                'skye terrier',
                'spinone italiano',
                'swedish vallhund',
                'tibetan spaniel',
                'toy fox terrier',
                'toy manchester terrier',
                'welsh terrier']
   
    equation01 = Landmarks_Calc(image_one,detector,predictor)

    index = 0
    for category in CATEGORIES:
        if category in Breed_lst:
            path=os.path.join(DATA_DIR,category) 
            print(path)
            for img in os.listdir(path):
                img_path = os.path.join(path,img)
                img_two=cv2.imread(img_path)
                

                count_match,len_keypoints_1, len_keypoints_2 = Feature_Extraction(image_one,img_two)
                data.loc[index,"Image2_Path"] = img_path
                data.loc[index,"Image_One_KeyPoints"] = len_keypoints_1
                data.loc[index,"Image_Two_KeyPoints"] = len_keypoints_2
                data.loc[index,"Count_Matches"] = count_match

                data.loc[index,"Landmarks_Input_Image"] = equation01
          
                find_index = landmark_df[landmark_df['Image2_Path']==img_path].index.tolist()
               
                
                landmark_series = landmark_df.loc[find_index,"Landmarks_Missing_Image"]
                landmark_index = landmark_series.index[0]
                landmark_value  = landmark_series[landmark_index]
                data.loc[index,"Landmarks_Missing_Image"] = landmark_value

                perc = calc_percentage_landmarks(len_keypoints_1,count_match)
                data.loc[index,"Feature_Percentage"] = perc

                minus = Difference_Landmarks(equation01,landmark_value)
                data.loc[index,"Landmark_Difference"] = minus

                land_per = Landmark_Percentage(minus)
                data.loc[index,"Percentage_Landmark"] = land_per

                target_sum = Target_Sum(perc,land_per)
                data.loc[index,"Target"] = target_sum

                index = index + 1
                print("index: ",index)
                print("data: ",data)

    return data

def return_top_3(df):
    df1=df.sort_values(by='Target',ascending=False)
    df1=df1[:3]
    response={'prediction':[]}
    for index,rows in df1.iterrows():
        response['prediction'].append({'Image2_Path':rows['Image2_Path'],'Target':rows['Target']})
        
       
    return response

def return_top3_score(lost_dict,df2):
    puid_list = []
    print(df2)
    if len(lost_dict) <=2:
        df1=df2.sort_values(by='Target',ascending=False)
        #print(df1)
        df3=df1.sort_values(by='Target',ascending=True)
    else:
        df1=df2.sort_values(by='Target',ascending=False)
        df1=df1[:3]
        #print(df1)
        df3=df1.sort_values(by='Target',ascending=True)

    # df1['Image2_Path'] --- avatar
    for index,rows in df3.iterrows():
        pid = rows['Image2_Path']
        print("Image2_Path: ",pid)
        a=pid.flatten()
        for key,value in lost_dict.items():
            b = value.flatten()
            if np.array_equal(a,b):
                print("Appending key: ",key)
                puid_list.append(key)
    print("Top 3 Matches: ",puid_list)
    response={'Top Match':[]}
    for index,rows in df1.iterrows():
        response['Top Match'].append({'PUID':puid_list[index],'Target':rows['Target']})
               
    return response

def crop_face(left,top,right,bottom):
    if left > 0:
        pass
    else:
        number = abs(left)
        print(number)
        left = left + number
    
    if top > 0:
        pass
    else:
        number = abs(top)
        print(number)
        top = top + number
        
    
    if right > 0:
        pass
    else:
        number = abs(right)
        print(number)
        right = right + number
        
    if bottom > 0:
        pass
    else:
        number = abs(bottom)
        print(number)
        bottom = bottom + number
        
        
    return left,top,right,bottom
    
    

def Detect_And_Crop_Face(img,detector):

    dets = detector(img, upsample_num_times=1)

    print("detector: ",len(dets))
    if(len(dets)>0):
                
        left=dets[0].rect.left()
        top=dets[0].rect.top()
        right=dets[0].rect.right()
        bottom=dets[0].rect.bottom()
    
        if (left > 0) and (right > 0) and (top > 0) and (bottom > 0):
                
                
            img=img[top:bottom,left:right]
            
            return ["Image",img]

        
        else:
            
            left,top,right,bottom = crop_face(left,top,right,bottom)
            
            img=img[top:bottom,left:right]

            
            return ["Image",img]

    else:
        return ["NotImage","NotImage"]

def string_to_list(st):
    new_st = st.split("[")[1]
    new_st_upd = new_st.split("]")[0]
    comma_values = new_st_upd.split(",")
    my_list = list()
    for i in range(len(comma_values)):
        value = comma_values[i].split("'")
        string_breed = value[1]
        my_list.append(string_breed)
        
    return my_list

def Feature_Match(image,db_path,detector,predictor):
    try:
        npimg = np.fromstring(image, np.uint8)
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        print("img",img)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print("Image rgb",img_rgb)
        my_list = Detect_And_Crop_Face(img,detector)
        status_img=my_list[0]
        img_crop=my_list[1]
        print(status_img)

        if status_img == "Image":
            
            cv2.imwrite("prediction_image.png",img_crop)

            breed_dict = Main_Processing_For_Identification(img_crop)
            string_list = breed_dict["breed"]
            breed_list=string_to_list(string_list)

            data = Match_in_DB(img_crop,db_path,breed_list,detector,predictor)
            
            data.to_excel("Output.xlsx")

            #result_df = data[["Image2_Path","Target"]]
            
            #result_df.sort_values('Target', inplace=True, ascending=False)
            
            #print(result_df)

            #dictionary = result_df.to_dict()
            dictionary =  return_top_3(data)
            print("Dictinary Output: ",dictionary)
            return dictionary

        else:
            dictionary =  {"msg": "Can't find proper face angle of the dog, Click and try again."}
            print("Dictinary Output: ",dictionary)
            return dictionary
            
    except Exception as e:
        print(e)
        return {"msg":"can not handle this image","problem":str(e)}
