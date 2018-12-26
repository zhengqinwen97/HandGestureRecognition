#%%
import os
# os.chdir("D:\\Code\\Graduation Project\\Gesture_detection_and_classify\\storePic\\palm1")

rootdir = "D:\\Code\\Graduation Project\\Gesture_detection_and_classify\\storePic\\palm1"
i = 0
for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        print(os.path.join(parent, filename))
        newName=str(i)+".jpg"
        os.rename(os.path.join(parent, filename), os.path.join(parent, newName))
        i = i+1

#%%
import tensorflow as tf
from skimage import io,transform
import glob
import os

import numpy as np
import time

#%%
path = "D:\\Code\Graduation Project\\Gesture_detection_and_classify\\storePic"
w=100
h=100
c=3

#%%
def read_img(path):
    # cate=[path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate=[path + "\\" + x for x in os.listdir(path) if os.path.isdir(path + "\\" + x)]
    # print(cate)
    imgs=[]
    labels=[]
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '\\*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data, label=read_img(path)
print(data)

#%%
#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]
print(label)
