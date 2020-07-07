import pyautogui
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

import pytesseract
#configure tesseract to use (English language, LSTM neural network, and single-line of text)
config = ("-l eng --oem 1 --psm 7")    # $ tesseract --help-oem 

def iamnothuman():
    #click i am human box
    #CLICK_BOX_POSITION = (453,410)
    pyautogui.click(453,410)

def get_category(img):
    '''
    param : img:screenshot of pahe.in
    return: the category to choose images from
    '''
    OCR_POSITION = (822,150,916,168)
    #extracting region of interest(ROI)
    roi = img.crop(OCR_POSITION)
    #extracting category
    category = pytesseract.image_to_string(roi, config=config)
    
    return category

def get_coordinates(initial = (493,299,631,437)):
    '''return a list of image coordinates in numpy array
    '''
    pic_coors1 = [np.array(initial)]
    pic_coors2 = []
    for i in range(1,3):
        new_pic = np.array(initial) + np.array((150,0,150,0))*i
        pic_coors1.append(new_pic)
        
    for img in pic_coors1:
        for i in range(1,3):
            new_pic = img + np.array((0,150,0,150))*i
            pic_coors2.append(new_pic)
    
    return pic_coors1 + pic_coors2

def extract_photos(img):
    '''
    param: img:screenshot image
    return : list of PIL IMAGE objects of 9 images
    '''
    pic_coors = get_coordinates()
    pics = [img.crop(coor) for coor in pic_coors]
    return pics

def preprocess_imgs(img_list):
    '''
    param: img_list: PIL IMAGE list from extract_photos func
    *********PIPELINE************
    1.resize images to (224,224) to pass into pretrained models
    2.scale data into (0,1) range
    3.convert to float tensors to pass into model (also note that tensors should be (channels,height,width) we will have to transpose images)
    4.normalize images with 
                            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                                             
    return : list with processed float tensors images                                                                
    '''
    processed_tensors = []
    for img in img_list:
        img = TF.resize(img,(224,224))
        img = TF.to_tensor(img)
        img = TF.normalize(img,
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        processed_tensors.append(img)
        
    return processed_tensors