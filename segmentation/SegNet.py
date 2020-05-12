import argparse
import cv2
import numpy as np
import keras
from PIL import Image
from keras.models import Model, load_model
from custom_layers import MaxPoolingWithIndices, UpSamplingWithIndices
import time

keras.backend.set_learning_phase(0)
class SegNet(object):
    def __init__(self, cfg):
        # load the model
        self.cfg = cfg
        self.model = load_model(self.cfg['segnet_model_path'],
            custom_objects={
                'MaxPoolingWithIndices': MaxPoolingWithIndices,
                'UpSamplingWithIndices': UpSamplingWithIndices
            },
            compile=False
        )
        self.avgimage = np.load(self.cfg['avgimage_path'])
        print("Initialized SegNet")
        
    def get_mask(self, imgs):
        for idx, img in enumerate(imgs):
            img = np.array(img.resize((320, 240), Image.BILINEAR))
            img = np.stack([img, img, img], axis=2)
            # subtract the training average
            img = img - self.avgimage
            imgs[idx] = img
            
        # predict with the model
        inbatch = np.stack(imgs, axis = 0)
        t1 = time.time()
        out = self.model.predict(inbatch)
        print("predict time =", time.time() - t1)
        masks = out[:,:,:,1]<0.5
        masks = masks.astype('uint8') * 255
        masks = [masks[i] for i in range(len(masks))]
        
        return masks
        
    def get_circle(self, mask):
        # find iris circle
        mask_for_iris = 255 - mask
        iris_indices = np.where(mask_for_iris[40:-40, 40:-40] == 0)
        iris_radius_estimate = max(max(iris_indices[0]) - min(iris_indices[0]), max(iris_indices[1]) - min(iris_indices[1])) // 2
        iris_circle = cv2.HoughCircles(mask_for_iris, cv2.HOUGH_GRADIENT, 1, 50,
                                       param1=self.cfg["iris_hough_param1"],
                                       param2=self.cfg["iris_hough_param2"],
                                       minRadius=iris_radius_estimate-self.cfg["iris_hough_lowermargin"],
                                       maxRadius=iris_radius_estimate+self.cfg["iris_hough_uppermargin"])
        iris_x, iris_y, iris_r = np.rint(np.array(iris_circle[0][0])*2).astype(int)
        
        # find pupil circle
        pupil_circle = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 50,
                                        param1=self.cfg["pupil_hough_param1"],
                                        param2=self.cfg["pupil_hough_param2"],
                                        minRadius=self.cfg["pupil_hough_minimum"],
                                        maxRadius=iris_r//2-self.cfg["pupil_hough_margin"])
        pupil_x, pupil_y, pupil_r = np.rint(np.array(pupil_circle[0][0])*2).astype(int)
        
        if np.sqrt((pupil_x-iris_x)**2+(pupil_y-iris_y)**2) > self.cfg["max_separation"]:
            pupil_x = iris_x
            pupil_y = iris_y
            pupil_r = iris_r // 3
        mask = cv2.resize(mask, (640, 480), interpolation = cv2.INTER_AREA)
        
        return mask, np.array([pupil_x, pupil_y,pupil_r]), np.array([iris_x, iris_y,iris_r])
        