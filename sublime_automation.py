import torch
from torchvision import models
import pyautogui
import time
import sublime_preprocess as pre

print('\n*****LOADING PRE-TRAINED MODEL*****')

#loading pretrained VGG16 model
vgg16 = models.vgg16(pretrained=True)
#set to evaluation mode as we are inferencing
vgg16.eval()

print('\n*****LOADING DONE*****\n')

#add a countdown to give some time
for _ in range(10,0,-1):
	print(f'Taking screenshot in {_+1} seconds')
	time.sleep(1)

pre.iamnothuman()

#take screenshot
im = pyautogui.screenshot()

pil_imgs = pre.extract_photos(im)
img_tensors = pre.preprocess_imgs(pil_imgs)

imagenet_idx2cls = torch.load('imagenet_idx2cls.pt')

#create a dictionary to map ocr to class labels in imagenet 
ocr_dict = {'airplane':['airliner', 'warplane, military plane', 'wing'],
            'a bus':['passenger car, coach, carriage', 'minibus', 'school bus', 'moving van','trolleybus, trolley coach, trackless trolley'],
            'umbrella':['umbrella', "academic gown, academic robe, judge's robe"],
            'umbrella:':['umbrella', "academic gown, academic robe, judge's robe"],
            'bicycle':['mountain bike, all-terrain bike, off-roader', 'bicycle-built-for-two, tandem bicycle, tandem','tricycle, trike, velocipede','mountain bike, all-terrain bike, off-roader','unicycle, monocycle']}


category = pre.get_category(im)
coors = pre.get_coordinates()

for e in range(2):
    for i in range(9): #iterate over each image
        scores = vgg16(img_tensors[i].unsqueeze(0))
        _, top_preds = torch.topk(scores,5,1)
        top_preds = [imagenet_idx2cls[pred.item()] for pred in top_preds[0]]

        for label in ocr_dict[category]:
            if label in top_preds:
                imge_coors = coors[i]
                x,y = (imge_coors[0] + imge_coors[2])/2, (imge_coors[1] + imge_coors[3])/2
                pyautogui.click(x,y,button='left')

                break

    pyautogui.click(889,797)
    if e==0:
        time.sleep(2)
        im2 = pyautogui.screenshot()
        pil_imgs2 = pre.extract_photos(im2)
        img_tensors = pre.preprocess_imgs(pil_imgs2)
