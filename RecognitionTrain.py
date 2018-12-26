#%%
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
from sklearn.utils import shuffle

#%%
# test the code
image = cv2.imread('handGesturePic/palm1/1.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
new_img = gray_image.reshape(256, 256, 1)
print(new_img.shape)

#%%
#Load Images from Palm1
loadedImages = []
for i in range(0, 600):
    image = cv2.imread('./handGesturePic/palm1/' + str(i) + '.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(256, 256, 1))

#Load Images From Palm2
for i in range(0, 600):
    image = cv2.imread('./handGesturePic/palm2/' + str(i) + '.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(256, 256, 1))
    
#Load Images From Fist
for i in range(0, 600):
    image = cv2.imread('./handGesturePic/fist/' + str(i) + '.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(256, 256, 1))

#%%
# Create OutputVector
outputVectors = []
for i in range(0, 600):
    outputVectors.append([1, 0, 0])

for i in range(0, 600):
    outputVectors.append([0, 1, 0])

for i in range(0, 600):
    outputVectors.append([0, 0, 1])


#%%
testImages = []

#Load Images for swing
for i in range(0, 200):
    image = cv2.imread('./handGesturePic/palm1_test/' + str(i) + '.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(256, 256, 1))

#Load Images for Palm
for i in range(0, 200):
    image = cv2.imread('./handGesturePic/palm2_test/' + str(i) + '.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(256, 256, 1))
    
#Load Images for Fist
for i in range(0, 200):
    image = cv2.imread('./handGesturePic/fist_test/' + str(i) + '.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(256, 256, 1))

testLabels = []

for i in range(0, 200):
    testLabels.append([1, 0, 0])
    
for i in range(0, 200):
    testLabels.append([0, 1, 0])

for i in range(0, 200):
    testLabels.append([0, 0, 1])

#%%
# Define the CNN Model
tf.reset_default_graph()
convnet=input_data(shape=[None,256,256,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,3,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

#%%
# Shuffle Training Data
loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)

# Train model
model.fit(loadedImages, outputVectors, n_epoch=50,
           validation_set = (testImages, testLabels),
           batch_size = 16, 
           snapshot_step=100, show_metric=True, run_id='convnet_coursera')

model.save(r"D:\Code\Graduation_Project\Gesture_detection_and_classify\new_models\GestureRecogModel.tfl")