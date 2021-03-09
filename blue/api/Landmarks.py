import pandas as pd
import os
import numpy as np
import cv2


dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-2]
dirname = "/".join(dirname_list)
landmark_path = dirname + "/Landmarks_Data.xlsx"

data = pd.read_excel(landmark_path)
data = data.drop("Unnamed: 0",axis=1)
data = data[["Image2_Path","Landmarks_Missing_Image","Breeds"]]

from Features import Landmarks_Calc
from Functions import Main_Processing_For_Identification

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


#def Update_Landmarks(image,breed,filename,db_path,detector,predictor):
def Update_Landmarks(image,filename,db_path,detector,predictor):
    try:
        index  = len(data)
        print(index)
        #img_path = db_path + "/" + breed + "/" + filename
        temp_img_path = db_path  + "/" + filename

        image.save(temp_img_path)
        img = cv2.imread(temp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        breed_dict = Main_Processing_For_Identification(img)
        print("breed dict: ",breed_dict)
        print(type(breed_dict))
        breed_list_add = breed_dict["breed"]
        print("breed list: ",breed_list_add)

        breed_list=string_to_list(breed_list_add)

        breed = breed_list[0]
        print("breed:",breed)
        
        img_path = db_path + "/" + breed + "/" + filename
        cv2.imwrite(img_path,img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        os.remove(temp_img_path)
        print("deleted")

        dets = detector(img, upsample_num_times=1)

        print("Length of detector",dets)
        if(len(dets)>0):
                    
            left=dets[0].rect.left()
            top=dets[0].rect.top()
            right=dets[0].rect.right()
            bottom=dets[0].rect.bottom()

            print(left,top,right,bottom)

        
            if (left > 0) and (right > 0) and (top > 0) and (bottom > 0):
                    
                    
                img=img[top:bottom,left:right]
                
                print("image saved in: ",img_path)
                cv2.imwrite(img_path,img)
                img = cv2.imread(img_path)
                
                # breed_dict = Main_Processing_For_Identification(img)
                # print("breed dict: ",breed_dict)
                # print(type(breed_dict))
                # breed_list_add = breed_dict["breed"]
                # print("breed list: ",breed_list_add)

                equation = Landmarks_Calc(img,detector,predictor)

                data.loc[index,"Image2_Path"] = img_path
                data.loc[index,"Landmarks_Missing_Image"] = equation
                data.loc[index,"Breeds"] = breed_list_add

                
                data.to_excel(landmark_path)
                
                return "Successfully Updated"

            
            else:
                
                left,top,right,bottom = crop_face(left,top,right,bottom)
                
                img=img[top:bottom,left:right]


                cv2.imwrite(img_path,img)
                img = cv2.imread(img_path)
                
                breed_dict = Main_Processing_For_Identification(img)
                print("breed dict: ",breed_dict)
                print(type(breed_dict))
                breed_list_add = breed_dict["breed"]
                print("breed list: ",breed_list_add)

                equation = Landmarks_Calc(img,detector,predictor)

                data.loc[index,"Image2_Path"] = img_path
                data.loc[index,"Landmarks_Missing_Image"] = equation
                data.loc[index,"Breeds"] = breed_list_add
                
                data.to_excel(landmark_path)
                
        
                return "Successfully Updated"

        else:
            os.remove(img_path)
            return "Can't find proper face angle of the dog, Click and try again."

    except Exception as e:
        print(e)
        os.remove(img_path)
        return f"Can't find proper face angle of the dog, Click and try again or problem is {e}"



