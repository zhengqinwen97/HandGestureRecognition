#%%
import cv2
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
from RecognitionLoadModel import load_my_model 
import time
import os

#%%
model = load_my_model(model_path = r"D:\Code\Graduation_Project\Gesture_detection_and_classify\new_models\GestureRecogModel.tfl")

pathdir = r"D:\Code\Graduation_Project\Gesture_detection_and_classify\001"

if os.path.exists(pathdir):
    os.removedirs(pathdir)

#%%
mydic = {0 : 'palm1', 1 : 'palm2', 2 : 'fist'}
while True:
    time.sleep(0.05)
    if not os.path.exists(pathdir):  
        try:
            os.mkdir(pathdir)
            image = cv2.imread(r'D:\Code\Graduation_Project\Gesture_detection_and_classify\Tmp.jpg')
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction = model.predict([gray_image.reshape(256, 256, 1)])
            predict_indx = np.argmax(prediction)
            print(mydic[predict_indx])

            # show tsxt in the image
            font=cv2.FONT_HERSHEY_SIMPLEX
            image=cv2.putText(image, mydic[predict_indx], (0,40), font, 1.2, (255,255,255), 2)  # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
            cv2.imshow('my_test_img', image.astype(np.uint8))
            
            # print("show local img")
            os.removedirs(pathdir)
        except FileExistsError:
            print("ERROR-------------------------------------------------->")
            pass             
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

if os.path.exists(pathdir):
    os.removedirs(pathdir)
