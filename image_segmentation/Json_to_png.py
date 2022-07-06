
import os 
import cv2  
import sys
import json
from PIL import Image, ImageDraw
import numpy as np
import math 
from PIL import Image, ImageDraw 
from PIL import ImagePath  
from tqdm import tqdm 


train_path = '/content/drive/MyDrive/idd-20k-II/idd20kII/leftImg8bit/train'  
val_path = '/content/drive/MyDrive/idd-20k-II/idd20kII/leftImg8bit/val' 
train_json_path = '/content/drive/MyDrive/idd-20k-II/idd20kII/gtFine/train'
val_json_path =  '/content/drive/MyDrive/idd-20k-II/idd20kII/gtFine/val'
trainlabelpath = '/content/drive/MyDrive/idd-20k-II/idd20kII/trainlabel/'
vallabelpath = 'content/drive/MyDrive/idd-20k-II/idd20kII/vallabel/'

colour = {'road':6,
'drivable fallback':6,
'motorcycle':1,
'sky':2,
'curb':3,
'building':0,
'vegetation':0,
'obs-str-bar-fallback':3,
'billboard':3,
'autorickshaw':1,
'pole':3,
'car':1,
'truck':1,
'person':4,'animal':4,
'rider':4,
'vehicle fallback':1,
'non-drivable fallback':5,
'wall':3,
'fallback background':2,
'fence':3,
'traffic sign':3,
'bus':1,
'bicycle':1,
'parking':6,
'bridge':0,
'tunnel':0,
'out of roi':7,
'ground':5,
'polegroup':3,
'sidewalk':5,
'guard rail':3,
'caravan':1,
'traffic light':3,
'trailer':1,
'rail track':5,
'ego vehicle':1,
'rectification border':5,
'unlabeled':7,
'train':1,
'license plate':1}

def get_poly_mask_fun(file):
    import json
    f = open(file, 'r')
    data = json.loads(f.read())
    global h,w
    h = data['imgHeight']
    w = data['imgWidth']
    for obj in data['objects']:
        label.append(obj['label'])
        poly.append(obj['polygon'])
        
def get_poly(index,mask):
    if mask == "Train":
        print("Processing:"+train_json[index])
        get_poly_mask_fun(train_json[index])
    elif mask == "Val":
        print("Processing:"+val_json[index])
        get_poly_mask_fun(val_json[index])            
    for k in range(len(poly)):
        vertex = []
        for i in range(len(poly[k])):
            vertex.append(tuple(poly[k][i]))
        vertexlist.append(vertex)


def convert_json_to_png():
    x_train = []
    print(colour)
    rootdir = train_path
    for root, dirs, files in os.walk(rootdir):
        for name in files:
            if name.endswith((".jpg")):
                x_train.append(os.path.join(root, name).replace('\\','/'))

    x_val = []
    rootdir = val_path
    for root, dirs, files in os.walk(rootdir):
        for name in files:
            if name.endswith((".jpg")):
                x_val.append(os.path.join(root, name).replace('\\','/'))

    train_json = []
    rootdir = train_json_path
    for root, dirs, files in os.walk(rootdir):
        for name in files:
            if name.endswith((".json")):
                train_json.append(os.path.join(root, name).replace('\\','/'))

    val_json = []
    rootdir = val_json_path
    for root, dirs, files in os.walk(rootdir):
        for name in files:
            if name.endswith((".json")):
                val_json.append(os.path.join(root, name).replace('\\','/'))

    outlayer_train = []
    count_train = 1
    
    val_name = []
    for path in val_json:
        val_name.append(path.split('/')[-2]+ '_' + path.split('/')[-1] )

    train_name = []
    for path in train_json:
        train_name.append(path.split('/')[-2]+ '_' + path.split('/')[-1] )

    for json in range(len(train_name)):
        label = []
        poly = []
        vertexlist = []
        get_poly(json,"Val")
        img = Image.new("L", (w, h))  
        img1 = ImageDraw.Draw(img)
        print(count_train)
        count_train = count_train + 1
        for j in range(len(vertexlist)):
            if len(vertexlist[j]) > 1:
                img1.polygon(vertexlist[j], fill = colour[label[j]])
        img.save(trainlabelpath+train_name[json].split('.')[0]+'.png', 'PNG')

    for json in range(len(val_name)):
        label = []
        poly = []
        vertexlist = []
        get_poly(json,"Val")
        img = Image.new("L", (w, h))  
        img1 = ImageDraw.Draw(img)
        print(count_val)
        count_val = count_val + 1
        for j in range(len(vertexlist)):
            if len(vertexlist[j]) > 1:
                img1.polygon(vertexlist[j], fill = colour[label[j]])
        img.save(vallabelpath+val_name[json].split('.')[0]+'.png', 'PNG')

if __name__ == "__main__":
   convert_json_to_png()
