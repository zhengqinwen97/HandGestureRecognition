from skimage import io,transform
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# continuons predict camera image
class PredictContinuons():
    def __init__(self, local_img):
        self.local_img = local_img[:]
        self.w = 100
        self.h = 100
        self.dic = {0:'fist', 1:'palm1', 2:'palm2'}
        
    def preprocess(self):
        img = self.local_img[:]
        img = transform.resize(img, (self.w, self.h))
        while True:
            cv2.imshow('img', img)
        # img.resize((self.w, self.h))
        return img

    def predictpic(self):
        with tf.Session() as sess:
            data = []
            data1 = self.preprocess()
            # self.local_img.resize((self.w, self.h))
            data.append(data1)

            saver = tf.train.import_meta_graph('./classify/modelSave/model.ckpt.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./classify/modelSave/'))

            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            feed_dict = {x : data}

            logits = graph.get_tensor_by_name("logits_eval:0")

            classification_result = sess.run(logits, feed_dict)
            return tf.argmax(classification_result, 1).eval()

if __name__ == "__main__":
    path1 = "./storePic/palm1.jpg"
    img = io.imread(path1)
    img_convert_ndarray = np.array(img)
    while True:
        cv2.imshow('img', img)
    # print(type(img)
    # print(img.shape)
    pred = PredictContinuons(img_convert_ndarray)
    print(pred.predictpic())