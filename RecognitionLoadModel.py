import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
from sklearn.utils import shuffle

def load_my_model(model_path):
    # Model defined
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

    tmp_model=tflearn.DNN(convnet,tensorboard_verbose=0)

    # Load Saved Model
    tmp_model.load(model_path)

    return tmp_model